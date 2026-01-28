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
 * Models a post-operand Operator
 * object e.g the !, inverse, square, cube operators
 *
 * @author GBEMIRO
 */
public class UnaryPostOperator extends Operator implements Validatable{


    /**
     * The precedence of this UnaryPostOperator object.
     */
   private final Precedence precedence;
    /**
     * The index of this operator
     * in the scanned Function that it belongs to.
     */
    private int index;
    /**
     * Creates a new UnaryPostOperator object
     *
     * @param name The name that identifies this UnaryPostOperator object
     * @param scan The List object that contains the tokens
     */
    public UnaryPostOperator(String name,int index,ArrayList<String>scan) {
      super( isUnaryPostOperator(name)?name:"");

        if(this.getName().equals("")){
            throw new IndexOutOfBoundsException("Invalid Name For Unary Post-Number Operator."  );
        }//end if
        else{
            this.index=(index>=0&&scan.get(index).equals(name))?index:-1;
                this.precedence= getPrecedence(name);
        }//end else

    if(this.index==-1){
                    throw new IndexOutOfBoundsException("Invalid Index"  );
    }

    }
/**
 *
 * @return the Precedence of this Operator object.
 */
    public Precedence getPrecedence() {
        return precedence;
    }














/**
 * @param scan the scanner-list object
 * that this UnaryPostOperator object exists in.
 * validates the grammatical usage of this operator (by leaving the correctFunction attribute of the function object un-modified)
 * if the usage of this operator
 * in its immediate environment i.e to its left and right is correct.
 * @return true if the grammatical usage of this token with respect to its 2 immediate neighboring
 * tokens to the left and to the right is correct.
 */
    @Override
public boolean validate(ArrayList<String>scan){
    boolean correct = true;
        try{
             //specify valid tokens that can come before a post-number operator
         if(!isNumber(scan.get(index-1))&&!isClosingBracket(scan.get(index-1))&&
          !isVariableString(scan.get(index-1))&&!isUnaryPostOperator(scan.get(index-1))
          ){
           util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index-1)+"\" And \""+scan.get(index)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector For Post-number operators!" );
             correct=false;
             scan.clear();
         }//end if
       //specify valid tokens that can come after a post-number operator
         if(!Method.isUnaryPreOperatorORDefinedMethod(scan.get(index+1))&&
     !isBinaryOperator(scan.get(index+1))&&!isUnaryPostOperator(scan.get(index+1))
          &&!Method.isLogToAnyBase(scan.get(index+1))&&!Method.isAntiLogToAnyBase(scan.get(index+1))
          &&!isClosingBracket(scan.get(index+1))
          &&!Method.isNumberReturningStatsMethod(scan.get(index+1))
          &&!isLogicOperator(scan.get(index+1))
              ){
             util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index)+"\" And \""+scan.get(index+1)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector Post-number operators!" );
             correct=false;
             scan.clear();
         }//end if
         }//end try
         catch(IndexOutOfBoundsException ind){

}//end catch

return correct;
}



    /**
     * Carefully interpretes the correct arrangement of a loose math statement
     * for objects of this class and applies the correct one to the
     * MathExpression object. Examples: ²,³,-¹,!
     *
     * @param scan The ArrayList object that is the scanner of the
     * MathExpression object and so contains this UnaryPostOperator object
     */
    public static void assignCompoundTokens(ArrayList<String> scan) {

        for (int i = 0; i < scan.size(); i++) {
            if (isUnaryPostOperator(scan.get(i))) {

                if (isNumber(scan.get(i - 1)) || isVariableString(scan.get(i - 1))) { //e.g. change: 9² to (9²) and 9²² to (9²²) etc.
                    int index = i - 1;
                    int j = i;
                    while (isUnaryPostOperator(scan.get(j))) {
                        ++j;
                    }
                    scan.add(j, ")");
                    scan.add(index, "(");
                    i = j + 1;
                }//end if
                else if (isClosingBracket(scan.get(i - 1))) {// e.g. sin,(,9,),²
                    int openIndex = Bracket.getComplementIndex(false, i - 1, scan);
                    int j = i;
                    while (isUnaryPostOperator(scan.get(j))) {
                        ++j;
                    }
                    if (Method.isDefinedMethod(scan.get(openIndex - 1))) {//check if the opening parenthesis is preceded by a function name
                        scan.add(j, ")");
                        scan.add(openIndex - 1, "(");
                    } else {
                        scan.add(j, ")");
                        scan.add(openIndex, "(");
                    }
                    i = j + 1;

                }

            }

        }


    }//end method










}//end class UnaryPostOperator
