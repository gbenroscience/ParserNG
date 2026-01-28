/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package parser;

import java.util.ArrayList;


/**
 *
 * @author GBENRO
 */
public class Operator{
/**
 * The name of the operator object
 */
protected String name;

 


    public static final String PLUS = "+";
    public static final String MINUS = "-";
    public static final String DIVIDE = "/";
    public static final String MULTIPLY = "*";
    public static final String EN_DASH = "–";
    public static final String FACTORIAL = "!";
    public static final String POWER = "^";
    public static final String LESS_THAN = "<";
    public static final String GREATER_THAN = ">";
    public static final String ASSIGN = "=";
    public static final String EQUALS = "==";
    public static final String LESS_OR_EQUALS = "≤";
    public static final String GREATER_OR_EQUALS = "≥";
    public static final String AND = "&";
    public static final String OR = "|";
    public static final String REMAINDER = "%";
    public static final String OPEN_CIRC_BRAC = "(";
    public static final String CLOSE_CIRC_BRAC = ")";
    public static final String COMMA = ",";
    public static final String ROOT = "√";
    public static final String CUBE_ROOT = "³√";
    public static final String PERMUTATION = "Р";
    public static final String COMBINATION = "Č";
    public static final String INVERSE = "-¹";
    public static final String SQUARE = "²";
    public static final String CUBE = "³";
    public static final String OPEN_SQUARE_BRAC = "[";
    public static final String CLOSE_SQUARE_BRAC = "]";
    public static final String COLON = ":";
    public static final String CONST = "const";
    public static final String STORE = "store:";
    public static final String EXIT = "exit:";
    public static final String SPACE = " ";
    public static final String SEMI_COLON = ";";
    public static final String AT = "@";






    /**
 * The instruction set of the
 * parser of this software.
 */
public static final String[] operators =
new String[]{PLUS,MINUS,DIVIDE,MULTIPLY,FACTORIAL,POWER,LESS_THAN,GREATER_THAN,
        ASSIGN,EQUALS,LESS_OR_EQUALS,GREATER_OR_EQUALS,AND,OR,REMAINDER,OPEN_CIRC_BRAC,CLOSE_CIRC_BRAC,COMMA,ROOT,CUBE_ROOT,
        PERMUTATION,COMBINATION,INVERSE,SQUARE,CUBE,OPEN_SQUARE_BRAC,CLOSE_SQUARE_BRAC,COLON,CONST,STORE,EXIT,SPACE,
        SEMI_COLON,AT
};


/**
 *
 * @param name creates a new form of a valid mathronian Operator
 */
public Operator(String name){
    this.name=name;
}
/**
 *
 * @param name set the name of the Operator object
 */
public void setName(String name){
    this.name=name;
}
/**
 *
 * @return the name of the Operator object
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
    return (op.equals(PLUS)||op.equals(MINUS)||op.equals(DIVIDE)||op.equals(MULTIPLY)
    ||op.equals(FACTORIAL)||op.equals(POWER)||op.equals(LESS_THAN)||op.equals(GREATER_THAN)||op.equals(ASSIGN)
    ||op.equals(EQUALS)||op.equals(GREATER_OR_EQUALS)||op.equals(LESS_OR_EQUALS)
    ||op.equals(AND)||op.equals(OR)||op.equals(REMAINDER)||op.equals(OPEN_CIRC_BRAC)||op.equals(CLOSE_CIRC_BRAC)
    ||op.equals(ROOT)||op.equals(CUBE_ROOT)||
     op.equals(COMBINATION)||op.equals(PERMUTATION)||op.equals(INVERSE)||op.equals(SQUARE)||op.equals(CUBE)||
     op.equals(COMMA))||op.equals(OPEN_SQUARE_BRAC)||op.equals(CLOSE_SQUARE_BRAC)||op.equals(COLON)||
     op.equals(CONST)||op.equals(STORE)||op.equals(EXIT)
     ||op.equals(SPACE)||op.equals(SEMI_COLON)||op.equals(AT);
}//end method isOperatorString

/**
 *
 * @param op the String object in consideration
 * @return true if the String object contains a exit command
 */
public static boolean isExitCommand(String op){
    return op.equals(EXIT);
}

public boolean isSemiColon(String op){
    return op.equals(SEMI_COLON);
}

public static boolean isAtOperator(String op){
return op.equals(AT);
}

/**
 *
 * @param op the String object in consideration
 * @return true if the String object contains a STORE command
 */
public static boolean isStoreCommand(String op){
    return op.equals(STORE);
}

/**
 *
 * @param op the String object in consideration
 * @return true if the String object contains a constant storage command
 */
public static boolean isConstantStoreCommand(String op){
    return op.equals(CONST);
}
/**
 *
 * @param op the String object in consideration
 * @return true if the String object represents a [ character
 */
public static boolean isOpeningBrace(String op){
    return op.equals(OPEN_SQUARE_BRAC);
}
/**
 *
 * @param op the String object in consideration
 * @return true if the String object represents a [ character
 */
public static boolean isClosingBrace(String op){
    return op.equals(CLOSE_SQUARE_BRAC);
}

/**
 *
 * @param op the String object in consideration
 * @return true if the String object represents a : character
 * This operator is used to mark the end-point of Variable initialization
 * e.g [A=2,AA=3,acy=8.838383]:3A+6AA-8acy
 * This instructs the parser to initialize the variables A, AA, acy with the values given and then evaluate the expression
 *
 */
public static boolean isColon(String op){
    return op.equals(COLON);
}
/**
 * @param op the String to check
 * @return true if the operator is a logic operator
 * the logic operators defined here are
 * ==,&lte;,&gte;,&lt;,&gt;,|,&amp;
 */
public static boolean isLogicOperator(String op){
   return (op.equals(EQUALS)||op.equals(LESS_OR_EQUALS)||op.equals(GREATER_OR_EQUALS)||op.equals(LESS_THAN)||
           op.equals(GREATER_THAN)||op.equals(OR)||op.equals(AND));
}

/**
 * @param op the String to check
 * @return true if the operator is the EQUALS operator
 */
public static boolean isEqualsOperator(String op){
    return ( op.equals(EQUALS) );
}
/**
 * @param op the String to check
 * @return true if the operator is the ASSIGN operator
 * it means that we assign or store the value of the RHS in the LHS
 * and so the LHS must represent a valid variable
 */
public static boolean isAssignmentOperator(String op){
    return ( op.equals(ASSIGN) );
}
/**
 * @param op the String to check
 * @return true if the operator is an operator that functions in between 2 numbers
 * or variables i.e,+,-,*,/,^,%,Č,Р
 */
public static boolean isBinaryOperator(String op){

return (op.equals(PLUS)||op.equals(MINUS)||op.equals(DIVIDE)||op.equals(MULTIPLY)
    ||op.equals(POWER)||op.equals(REMAINDER)||op.equals(COMBINATION)||op.equals(PERMUTATION) );
}


/**
 * @param op the String to check
 * @return true if the operator is the + operator or is the - operator
 * the form is log-¹(num,base)
 */
public static boolean isPlusOrMinus(String op){
    return ( op.equals(PLUS)||op.equals(MINUS) );
}
/**
 * @param op the String to check
 * @return true if the operator is the / operator or is the * operator
 * the form is log-¹(num,base)
 */
public static boolean isMulOrDiv(String op){
    return ( op.equals(DIVIDE)||op.equals(MULTIPLY) );
}
/**
 * @param op the String to check
 * @return true if the operator is the permutation operator or is the combination operator
 * the form is log-¹(num,base)
 */
public static boolean isPermOrComb(String op){
    return ( op.equals(COMBINATION)||op.equals(PERMUTATION) );
}
/**
 * @param op the String to check
 * @return true if the operator is the + operator or is the - operator
 * the form is log-¹(num,base)
 */
public static boolean isMulOrDivOrRemOrPermOrCombOrPow(String op){
    return ( isBinaryOperator(op)&&!isPlusOrMinus(op) );
}
/**
 * @param op the String to check
 * @return true if the operator is the REMAINDER operator
 */
public static boolean isRemainder(String op){
    return (op.equals(REMAINDER));
}
/**
 * @param op the String to check
 * @return true if the operator is the POWER operator
 */
public static boolean isPower(String op){
    return (op.equals(POWER));
}
/**
 * @param op the String to check
 * @return true if the operator is the OPEN_CIRC_BRAC  or the CLOSE_CIRC_BRAC operator
 */
public static boolean isBracket(String op){
return ( op.equals(OPEN_CIRC_BRAC)||op.equals(CLOSE_CIRC_BRAC));
}
/**
 * @param op the String to check
 * @return true if the operator is the OPEN_CIRC_BRAC operator
 */
public static boolean isOpeningBracket(String op){
return op.equals(OPEN_CIRC_BRAC);
}
/**
 * @param op the String to check
 * @return true if the operator is the CLOSE_CIRC_BRAC operator
 */
public static boolean isClosingBracket(String op){
return op.equals(CLOSE_CIRC_BRAC);
}
/**
 * @param op the String to check
 * @return true if the operator is the FACTORIAL operator
 */
public static boolean isFactorial(String op){
return op.equals(FACTORIAL);
}
/**
 * @param op the String to check
 * @return true if the operator is the INVERSE operator
 */
public static boolean isInverse(String op){
return op.equals(INVERSE);
}
/**
 * @param op the String to check
 * @return true if the operator is the ROOT operator
 */
public static boolean isSquareRoot(String op){
return op.equals(ROOT);
}
    /**
     * @param op the String to check
     * @return true if the operator is the "³√" operator
     */
    public static boolean isCubeRoot(String op){
        return op.equals(CUBE_ROOT);
    }
/**
 * @param op the String to check
 * @return true if the operator is the SQUARE operator
 */
public static boolean isSquare(String op){
return op.equals(SQUARE);
}
/**
 * @param op the String to check
 * @return true if the operator is the CUBE operator
 */
public static boolean isCube(String op){
return op.equals(CUBE);
}
/**
 * @param op the String to check
 * @return true if the operator is a pre-number operator
 * e.g the trig operators,exponential operators,logarithmic operators(not to any base)
 */
public static boolean isUnaryPreOperator(String op){
return ( op.equals(ROOT) || op.equals(CUBE_ROOT));
}
/**
 * @param op the String to check
 * @return true if the operator is a post number operator
 * e.g the inverse operator,the factorial,the square and the cube
 */
public static boolean isUnaryPostOperator(String op){
   return (op.equals(INVERSE)||op.equals(FACTORIAL)||op.equals(SQUARE)||op.equals(CUBE));
}

/**
 * @param op the String to check
 * @return true if the operator is the comma(,) operator
 */
public static boolean isComma(String op){
    return ( op.equals(COMMA) );
}




/**
 * The precedence of the operators
 * @param name the name of the Operator object
 * @return the Operator's Precedence attribute
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
    else if(isMulOrDiv(name)){
         return new Precedence(9997);
    }
    else if(isRemainder(name)){
        return new Precedence(9996);
    }
    else if(isPlusOrMinus(name)){
         return new Precedence(9995);
    }


return null;
}
 
/**
 *
 * @param scan An ArrayList object containing a
 * scanned function.
 * @return true if validated
 */
public static boolean validateAll(ArrayList<String>scan){
boolean correct=true;
    for(int i=0;i<scan.size();i++){
if(isBinaryOperator(scan.get(i))){
correct = new BinaryOperator(scan.get(i), i, scan).validate(scan);
}
else if(isLogicOperator(scan.get(i))){
correct = new LogicOperator(scan.get(i), i, scan).validate(scan);
}
 else if(isUnaryPostOperator(scan.get(i))){
correct = new UnaryPostOperator(scan.get(i), i, scan).validate(scan);
}
 else if(isUnaryPreOperator(scan.get(i))){
correct = new UnaryPreOperator(scan.get(i), i, scan).validate(scan);
}
    }//end for

return correct;
}//end method





}//end method