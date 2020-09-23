/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package parser;

import java.util.ArrayList;

/**
 *
 * @author GBEMIRO
 */
public final class CombinationOperator extends BinaryOperator{


   
//Č

/**
 * Creates an object of class
 * CombinationOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan the List object in which this
 * CombinationOperator exists.
 *
 */
    public CombinationOperator( int index,ArrayList<String>scan){
        super(COMBINATION, index,scan);
    }//end constructor


   

}
