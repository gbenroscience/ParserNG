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
 * Deals with number returning statistical operators
 * e.g sum,avg,mode e.t.c and log and antilog
 * to any base operators
 *
 * @author GBEMIRO
 */
public class NumberReturningStatsOperator extends Operator implements Validatable{
    /**
     * The index of this operator
     * in the scanned Function that it belongs to.
     */
    private int index;
    /**
     *
     * @param name The name of this Operator with which it is represented in a math
     * function.
     * @param index the location of this Operator object in its parent Function
     * object's scanned ArrayList object.
     * @param scan The scanner output
     */
    public NumberReturningStatsOperator(String name,int index,ArrayList<String>scan) {
        super((Method.isNumberReturningStatsMethod(name)|| Method.isLogOrAntiLogToAnyBase(name))?name:"");
        this.index=index;
        if(this.getName().equals("")){
            throw new IndexOutOfBoundsException("Invalid Name For Number Returning Statistical Operator."  );
        }//end if
      else{
            this.index=(index>=0&&scan.get(index).equals(name))?index:-1;
        }//end else

    if(this.index==-1){
                    throw new IndexOutOfBoundsException("Invalid Index"  );
    }

    }//end constructor


/**
 *
 * @param index sets the index of this operator
 * in the scanned Function that it belongs to.
 * @param scan The scanner output
 */
    public void setIndex(int index,ArrayList<String>scan) {

            this.index=(index>=0&&scan.get(index).equals(this.getName()))?index:-1;

    if(this.index==-1){
                    throw new IndexOutOfBoundsException("Invalid Index"  );
    }
    }
/**
 *
 * @return the index of this operator
 * in the scanned Function that it belongs to.
 */
    public int getIndex() {
        return index;
    }



/**
 * @param scan the ArrayList object
 * that this NumberReturningStatsOperator object exists in.
 * validates the grammatical usage of this operator 
 * (by leaving the correctFunction attribute of the Function object un-modified)
 * if the usage of this operator
 * in its immediate environment i.e to its left and right is correct.
 * @return true if valid
 */
    @Override
public boolean validate(ArrayList<String>scan){
boolean correct = true;

        //Number returning stats operators
        try{
             //specify valid tokens that can come before a number returning stats operator
        if(!isNumber(scan.get(index-1))&&!isBinaryOperator(scan.get(index-1))&&
          !isVariableString(scan.get(index-1))&&!isUnaryPostOperator(scan.get(index-1))
       &&!isLogicOperator(scan.get(index-1))&&!isAssignmentOperator(scan.get(index-1))
                 &&!Method.isAntiLogToAnyBase(scan.get(index-1))
                 &&!Method.isLogToAnyBase(scan.get(index-1))&&!isUnaryPreOperator(scan.get(index-1))
                 &&!isOpeningBracket(scan.get(index-1))
          ){
          util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index-1)+"\" And \""+scan.get(index)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector For Number Returning Stats operators!" );
             correct = false;
             scan.clear();
         }//end if
       //specify valid tokens that can come after a number returning stats operator
         if(!isOpeningBracket(scan.get(index+1))&&!isVariableString(scan.get(index+1))
         &&!Method.isLogToAnyBase(scan.get(index+1))&&!Method.isAntiLogToAnyBase(scan.get(index+1))
         &&!isUnaryPreOperator(scan.get(index+1))&&!isNumber(scan.get(index+1))
         ){
        util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index)+"\" And \""+scan.get(index+1)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector For Number Returning Stats operators!" );
             correct = false;
             scan.clear();
         }//end if
         }//end try
         catch(IndexOutOfBoundsException ind){

}//end catch
return correct;
}//end method validateNumberReturningStatsOperator.






}//end class NumberReturningStatsOperator
