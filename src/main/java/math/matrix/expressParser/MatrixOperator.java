/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import parser.MathExpression;
import parser.Precedence;
import parser.UnaryPreOperator;


import java.util.ArrayList;


/**
 *
 * @author GBENRO
 */
public class MatrixOperator{
/**
 * The name of the operator object
 */
private String name;


/**
 *
 * @param name creates a new form of a valid mathronian MOperator
 */
public MatrixOperator(String name){
    this.name=name;
}
/**
 *
 * @param name set the name of the MOperator object
 */
public void setName(String name){
    this.name=name;
}
/**
 *
 * @return the name of the MOperator object
 */
public String getName(){
    return name;
}



/**
 *
 * @param op The string to check.
 * @return true if the operator is a valid mathronian operator
 */
public static boolean isOperatorString(String op){

    return ( op.equals("+")||op.equals("-")||op.equals("*")
     ||op.equals("^")||op.equals("=") ||op.equals("(")||op.equals(")")
     ||op.equals("rowJoin")||op.equals("colJoin")||op.equals("tri")||
     op.equals("|")||op.equals("unit")||op.equals("det")||op.equals(":")||
     op.equals("-¹")||op.equals("²")||op.equals("³")||op.equals(",")
     ||op.equals("[")||op.equals("]") );

}//end method isOperatorString

/**
 *
 * @param op the String object in consideration
 * @return true if the String object contains a exit command
 */
public static boolean isRowJoin(String op){ 
    return op.equals("rowJoin");
}

/**
 *
 * @param op the String object in consideration
 * @return true if the String object contains a exit command
 */
public static boolean isColJoin(String op){
    return op.equals("colJoin");
}

public boolean isColon(String op){
    return op.equals(":");
}



/**
 *
 * @param op the String object in consideration
 * @return true if the String object contains a Variable storage command
 */
public static boolean isTri(String op){
    return op.equals("tri");
}
/**
 *
 * @param op the String object in consideration
 * @return true if the String object contains a constant storage command
 */
public static boolean isUnit(String op){
    return op.equals("unit");
}
/**
 * 
 * @param op the String object in consideration
 * @return true if the String object represents a [ character
 */
public static boolean isOpeningBrace(String op){
    return op.equals("[");
}
/**
 *
 * @param op the String object in consideration
 * @return true if the String object represents a [ character
 */
public static boolean isClosingBrace(String op){
    return op.equals("]");
}


/**
 * @param op the String to check
 * @return true if the operator is the "=" operator
 * it means that we assign or store the value of the RHS in the LHS
 * and so the LHS must represent a valid variable
 */
public static boolean isAssignmentOperator(String op){
    return ( op.equals("=") );
}
/**
 * @param op the String to check
 * @return true if the operator is an operator that functions in between 2 numbers
 * or variables i.e,+,-,*,/,^,%,Č,Р
 */
public static boolean isBinaryOperator(String op){

return (op.equals("+")||op.equals("-")||op.equals("*")||op.equals("^")
        ||op.equals("rowJoin")||op.equals("colJoin"));
}


/**
 * @param op the String to check
 * @return true if the operator is the + operator or is the - operator
 * the form is log-¹(num,base)
 */
public static boolean isPlusOrMinus(String op){
    return ( op.equals("+")||op.equals("-") );
}
/**
 * @param op the String to check
 * @return true if the operator is the * operator
 * the form is log-¹(num,base)
 */
public static boolean isMul(String op){
    return ( op.equals("*") );
}

/**
 * @param op the String to check
 * @return true if the operator is the "%" operator
 */
public static boolean isPower(String op){
    return (op.equals("^"));
}
/**
 * @param op the String to check
 * @return true if the operator is the "("  or the ")" operator
 */
public static boolean isBracket(String op){
return ( op.equals("(")||op.equals(")"));
}
/**
 * @param op the String to check
 * @return true if the operator is the "(" operator
 */
public static boolean isOpeningBracket(String op){
return op.equals("(");
}
/**
 * @param op the String to check
 * @return true if the operator is the ")" operator
 */
public static boolean isClosingBracket(String op){
return op.equals(")");
}
/**
 * @param op the String to check
 * @return true if the operator is the "-¹" operator
 */
public static boolean isInverse(String op){
return op.equals("-¹");
}
/**
 * @param op the String to check
 * @return true if the operator is the "|" operator
 */
public static boolean isDetHalfSymbol(String op){
return op.equals("|");
}
/**
 * @param op the String to check
 * @return true if the operator is the "|" operator
 */
public static boolean isDet(String op){
return op.equals("det");
}
/**
 * @param op the String to check
 * @return true if the operator is the "²" operator
 */
public static boolean isSquare(String op){
return op.equals("²");
}
/**
 * @param op the String to check
 * @return true if the operator is the "³" operator
 */
public static boolean isCube(String op){
return op.equals("³");
}
/**
 * @param op the String to check
 * @return true if the operator is a pre-number operator
 * e.g the trig operators,exponential operators,logarithmic operators(not to any base)
 */
public static boolean isUnaryPreOperator(String op){
return ( op.equals("det") ||op.equals("tri") ||op.equals("unit"));
}
/**
 * @param op the String to check
 * @return true if the operator is a post number operator
 * e.g the inverse operator,the factorial,the square and the cube
 */
public static boolean isUnaryPostOperator(String op){
   return (op.equals("-¹")||op.equals("²")||op.equals("³"));
}


/**
 * The precedence of the operators
 * @param name the name of the MOperator object
 * @return the MOperator's Precedence attribute
 */
public static Precedence getPrecedence(String name){

    if(isUnaryPostOperator(name)){
         return new Precedence(10000);
    }
    else if(isPower(name)){
        return new Precedence(9999);
    }
    else if(isUnaryPreOperator(name)){
        return new Precedence(9998);
    }

    else if(isMul(name)){
         return new Precedence(9997);
    }
    else if(isPlusOrMinus(name)){
         return new Precedence(9995);
    }


return null;
}


/*
 * Recognizes the compound tokens
 * in a math expression.
 * Compound tokens are groups of tokens that
 * have a particular meaning and so must be treated accordingly.
 * e.g 3/4sin2 is not the same as 3/4*sin2
 *
 *
 */
public static void orderCompoundTokens(MathExpression function){
ArrayList<String>scanner=function.getScanner();

for(int i=0;i<scanner.size();i++){
    if( isPower(scanner.get(i)) ){

        
    }//end if





}//end for









}

/**
 * 
 * @param scan An ArrayList object containing a 
 * scanned function.
 */
public static boolean validateAll(ArrayList<String>scan){
boolean correct=true;
    for(int i=0;i<scan.size();i++){

if(isUnaryPostOperator(scan.get(i))){
correct = new MUnaryPostOperator(scan.get(i), i, scan).validate(scan);
}
 else if(isUnaryPreOperator(scan.get(i))){
correct = new UnaryPreOperator(scan.get(i), i, scan).validate(scan);
}
    }//end for

return correct;
}//end method







}
