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
 * object e.g the trigonometric,logarithmic,exponential e.t.c. operators.
 *
 * @author GBEMIRO
 */
public final class UnaryPreOperator extends Operator implements Validatable{
    /**
     * The precedence of this BinaryOperator object.
     */
   private final Precedence precedence;


    /**
     * The index of this operator
     * in the scanned MathExpression that it belongs to.
     */
    private int index;
    /**
     * Creates a new UnaryPreOperator object
     * 
     * @param name The name of the operator
     * @param index The index of the operator in the list
     * @param scan The scanner output
     */
    public UnaryPreOperator(String name,int index,ArrayList<String>scan) {
      super( isUnaryPreOperator(name)?name:"");
        if(this.getName().equals("")){
            throw new IndexOutOfBoundsException("Invalid Name For Unary Pre-Number Operator."  );
        }//end if
        else{
   
            this.index=(index>=0&&scan.get(index).equals(name))?index:-1;
            this.precedence= getPrecedence(name);
        }//end else

    if(this.index==-1){
                    throw new IndexOutOfBoundsException("Invalid Index"  );
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
 * @param scan the ArrayList object
 * that this UnaryPreOperator object exists in.
 * validates the grammatical usage of this operator (by leaving the correctFunction attribute of the function object un-modified)
 * if the usage of this operator
 * in its immediate environment i.e to its left and right is correct.
 * @return true if the grammatical usage of this token with repect to its 2 immediate neighbouring
 * tokens to the left and to the right is correct.
 */
    @Override
public boolean validate(ArrayList<String>scan){

boolean correct = true;
    try{
           //specify valid tokens that can come before a pre-number operator
          if(!isBinaryOperator(scan.get(index-1))&&!isNumber(scan.get(index-1))&&
          !isUnaryPreOperator(scan.get(index-1))&&!isBracket(scan.get(index-1))&&
          !isVariableString(scan.get(index-1))&&!isUnaryPostOperator(scan.get(index-1))&&
          !isLogicOperator(scan.get(index-1))
          &&!isAssignmentOperator(scan.get(index-1))
          ){
                util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index-1)+"\" And \""+scan.get(index)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector For Pre-number operators!" );
             correct = false;
             scan.clear();
         }//end if
       //specify valid tokens that can come after a pre-number operator
         if(!isOpeningBracket(scan.get(index+1))&&!isNumber(scan.get(index+1))
           &&!Method.isUnaryPreOperatorORDefinedMethod(scan.get(index+1))&&
          !isVariableString(scan.get(index+1))&&!Method.isNumberReturningStatsMethod(scan.get(index+1))
          &&!Method.isLogToAnyBase(scan.get(index+1))&&!Method.isAntiLogToAnyBase(scan.get(index+1))
          ){
          util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index)+"\" And \""+scan.get(index+1)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector For Pre-number operators!" );
             correct = false;
            scan.clear();
         }//end if

         }//end try
         catch(IndexOutOfBoundsException ind){

}//end catch


return correct;
}



/**
 * 
 * @param function  the MathExpression object that contains the power operator
 */
public static void assignCompoundTokens(MathExpression function){
    ArrayList<String>scan=function.getScanner();



}











}//end class UnaryPreOperator
