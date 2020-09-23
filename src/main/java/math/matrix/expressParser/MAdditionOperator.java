/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import java.util.ArrayList;

/**
 *
 * @author GBEMIRO
 */
public final class MAdditionOperator extends MBinaryOperator {

//Č

/**
 * Creates an object of class
 * MAdditionOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan The scanner output
 */ 
    public MAdditionOperator(int index, ArrayList<String>scan){
        super("+", index, scan);
    }//end constructor


   

}//end class MAdditionOperator
