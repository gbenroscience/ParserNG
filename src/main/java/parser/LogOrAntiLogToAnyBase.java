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

public class LogOrAntiLogToAnyBase extends Operator implements Validatable{
/**
 * The index of objects of this
 * class in the scanned Function object
 * in which they exist.
 */
private int index;
    /**
     *
     * @param name The name of this Operator with which it is represented in a math
     * function.
     * @param index the location of this Operator object in its parent Function
     * object's scanned ArrayList object.
     * @param scan The scanned output
     */
    public LogOrAntiLogToAnyBase(String name,int index,ArrayList<String>scan) {
        super((Method.isLogOrAntiLogToAnyBase(name))?name:"");
        this.index=index;
        if(this.getName().equals("")){
            throw new IndexOutOfBoundsException("Invalid Name For Log or antilog to any base type Operator."  );
        }//end if
      else{
            this.index=(index>=0&&scan.get(index).equals(name))?index:-1;
        }//end else
    if(this.index==-1){
                    throw new IndexOutOfBoundsException("Invalid Index"  );
    }//end if
    }//end constructor
/**
 *
 * @param index sets the index
 */
    public void setIndex(int index) {
        this.index = index;
    }

    public int getIndex() {
        return index;
    }

 
    /**
     * 
     * @param scan The scanner output
     * @return true if valid
     */
    @Override
public boolean validate(ArrayList<String>scan){
boolean correct=true;
    try{

       //specify valid tokens that can come before a LogOrAntiLogToAnyBase operator
   if(!isNumber(scan.get(index-1))&&!isBinaryOperator(scan.get(index-1))&&
           !isVariableString(scan.get(index-1))&&!isBracket(scan.get(index-1))
           ){
             util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index-1)+"\" And \""+scan.get(index)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector For LogOrAntiLogToAnyBase Operators!" );
             correct=false;  scan.clear();
         }//end if
       //specify valid tokens that can come after a LogOrAntiLogToAnyBase operator
         if(!isNumber(scan.get(index+1))&&!isOpeningBracket(scan.get(index+1))&&!Method.isNumberReturningStatsMethod(scan.get(index+1))
            &&!isUnaryPreOperator(scan.get(index+1))
            &&!Method.isLogToAnyBase(scan.get(index+1))&&!Method.isAntiLogToAnyBase(scan.get(index+1))&&!Method.isStatsMethod(scan.get(index+1))
                 ){
            util.Utils.logError(
            "ParserNG Does Not Allow "+getName()+" To Combine The Function Members \""+scan.get(index)+"\" And \""+scan.get(index+1)+"\""+
                        " As You Have Done."+
            "ParserNG Error Detector For LogOrAntiLogToAnyBase Operators!" );
            correct=false;  scan.clear();
         }//end if
         }//end try
         catch(IndexOutOfBoundsException ind){

}//end catch
return correct;
}//end method validateLogOrAntiLogToAnyBase












}//end class
