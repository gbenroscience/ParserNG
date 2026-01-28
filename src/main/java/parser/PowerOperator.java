/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package parser;

import parser.methods.Method;

import java.util.ArrayList;
import static parser.Number.*;


/**
 *
 * @author GBEMIRO
 */
public final class PowerOperator extends BinaryOperator{


/**
 * PRECEDENCE VALUES:
 * ^
 * Р and Č
 * * and -
 * %
 * + and -
 * <
 * <=
 * >
 * >=
 * &
 * |
 *
 *
 *
 */


    /**
     * Creates an object of class
     * PowerOperator
     *
     * @param index the index of this object in its
     * parent Function object's scanner.
     * @param scan the Function object in which this
     * PowerOperator exists.
     *
     */
    public PowerOperator( int index,ArrayList<String>scan){
        super("^", index, scan);

    }//end constructor






    /**
     *
     * @param scan the ArrayList object that is the scanner and so contains the PowerOperator object.
     */
    public static void assignCompoundTokens(ArrayList<String>scan){

        /**
         * We have 2 possible scenarios.
         * One is for operations like 3^6,A^B i.e in general...[number|variable]^[number|variable]
         * or(expr)^(expr)
         *
         * The second is a bit more complex.i.e 3^sin(2^3).
         *
         * The first one must be evaluated throughout the Function object before the second one is taken up.
         * The first for loop takes care of the first scenario by checking if the right hand operand
         * is a number or a bracketed expression first and then if true applying the production. Else,
         * it does not apply the production.
         *
         */
        for(int i=0;i<scan.size();i++){
            try{
                if(isPower(scan.get(i))){
                    //variable that checks if the scenario is of type simple
                    //or type complex.
                    boolean isSimpleNumberOperandCase=false;
                    if(isOpeningBracket(scan.get(i+1))){
                        int j=i+1;
                        ///handle naughty user cases where the user
                        //enters the pattern 3^(((((sin....
                        while(isOpeningBracket(scan.get(j))){
                            ++j;
                        }//emerge from (((( pattern.
                        if(isNumber(scan.get(j+1))|| ( Variable.isVariableString(scan.get(j+1))&&!isOpeningBracket(scan.get(j+2)) )  ){
                            isSimpleNumberOperandCase=true;
                        }
                    }
                    else if(isNumber(scan.get(i+1))|| ( Variable.isVariableString(scan.get(i+1))&&!isOpeningBracket(scan.get(i+2)) ) ){
                        isSimpleNumberOperandCase=true;
                    }
//end of checking the scenario.The boolean isSimpleNumberOperandCase contains the scenario.
                    //if true, then we have scenario 1 and we apply the production now.
                    //else we do not apply the production.


                    if(isSimpleNumberOperandCase){//cos(x)^2
                        //deal with the elements to the PowerOperator's immediate left
                        if(isNumber(scan.get(i-1))|| ( Variable.isVariableString(scan.get(i-1))&&!isOpeningBracket(scan.get(i)) )  ){
                            scan.add(i-1,"(");
                            ++i;//The addition operation will cause elements to the right to shift to the right by one
                            //so keep track of the index of the power operator by incrementing its former index.
                        }
                        else if(isClosingBracket(scan.get(i-1))){
                            int index=Bracket.getComplementIndex(false, i-1, scan);

                            if( index-1 > 0 && Method.isMethodName(scan.get(index-1))){
                                continue;
                            }
                            else{
                                scan.add(index,"(");
                                ++i;//The addition operation will cause elements to the right to shift to the right by one
                                //so keep track of the index of the power operator by incrementing its former index.
                            }
                        }
                        //deal with the elements to the PowerOperator's immediate right
                        if(isNumber(scan.get(i+1))|| ( Variable.isVariableString(scan.get(i+1))&&!isOpeningBracket(scan.get(i+2)) ) ){
                            scan.add(i+2,")");
                            ++i;//skip to the point where the bracket was inserted, to avoid an infinite loop.
                        }//close up the bracket found when a number or variable was found to the left of the power operator.
                        else if(isOpeningBracket(scan.get(i+1))){
                            int index=Bracket.getComplementIndex(true, i+1, scan);
                            scan.add(index,")");
                            ++i;//skip to avoid an infinite loop.
                        }//close up the bracket opened when a close bracket was found to the immediate left of the power operator.

                    }//end if isSimpleNumberOperandCase

                }//end if isPower

            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }//end catch

        }//end for loop

        for(int i=0;i<scan.size();i++){
            try{
                if(isPower(scan.get(i))){
                    boolean isSimpleNumberOperandCase=false;
                    if(isOpeningBracket(scan.get(i+1))){
                        int j=i+1;
                        ///handle naughty user cases where the user
                        //enters the pattern 3^(((((sin....
                        while(isOpeningBracket(scan.get(j))){
                            ++j;
                        }
                        if(isNumber(scan.get(j+1))|| ( Variable.isVariableString(scan.get(j+1))&&!isOpeningBracket(scan.get(j+2)) )   ){
                            isSimpleNumberOperandCase=true;
                        }
                    }
                    else if(isNumber(scan.get(i+1))|| ( Variable.isVariableString(scan.get(i+1))&&!isOpeningBracket(scan.get(i+2)) ) ){
                        isSimpleNumberOperandCase=true;
                    }

//end of checking the scenario.The boolean isSimpleNumberOperandCase contains the scenario.
                    //if true, then we have scenario 1 and we apply the production now.
                    //else we do not apply the production.

                    if(!isSimpleNumberOperandCase){

                        //deal with the elements to the PowerOperator's immediate left
                        if(isNumber(scan.get(i-1))|| ( Variable.isVariableString(scan.get(i-1))&&!isOpeningBracket(scan.get(i)) )  ){
                            scan.add(i-1,"(");
                            ++i;//The addition operation will cause elements to the right to shift to the right by one
                            //so keep track of the index of the power operator by incrementing its former index.
                        }
                        else if(isClosingBracket(scan.get(i-1))){
                            int index=Bracket.getComplementIndex(false, i-1, scan);
                            if( index-1 > 0 && Method.isMethodName(scan.get(index-1))){
                                continue;
                            }
                            else{
                                scan.add(index,"(");
                                ++i;//The addition operation will cause elements to the right to shift to the right by one
                                //so keep track of the index of the power operator by incrementing its former index.
                            }
                        }
                        //deal with the elements to the PowerOperator's immediate right
                        //close up the bracket opened when a close bracket was found to the immediate left of the power operator.
                        if(isOpeningBracket(scan.get(i+1))){
                            int index=Bracket.getComplementIndex(true, i+1, scan);
                            scan.add(index,")");
                            ++i;//skip to avoid an infinite loop.
                        }

                        else if( isUnaryPreOperator(scan.get(i+1))||( Method.isDefinedMethod(scan.get(i+1))&&isOpeningBracket(scan.get(i+2)) ) ){
                            int j=i+1;
                            while( isUnaryPreOperator(scan.get(j))||( Method.isDefinedMethod(scan.get(j))&&isOpeningBracket(scan.get(j+1)) ) ){
                                ++j;
                            }
                            if(isNumber(scan.get(j))|| Variable.isVariableString(scan.get(j))){
                                scan.add(j+1, ")");
                                ++i;
                            }
                            else if(isOpeningBracket(scan.get(j))){
                                int index=Bracket.getComplementIndex(true, j, scan);
                                scan.add(index,")");
                                ++i;//skip to avoid an infinite loop.

                            }
                        }//end else if

                        else if(Method.isNumberReturningStatsMethod(scan.get(i+1))|| Method.isLogOrAntiLogToAnyBase(scan.get(i+1))){
                            int index=Bracket.getComplementIndex(true, i+2, scan);
                            scan.add(index,")");
                            ++i;
                        }






                    }


                }
            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }//end catch
        }//end for

    }//end method assignCompoundTokens


}
