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
public final class SubtractionOperator extends BinaryOperator{
  
//Č

/**
 * Creates an object of class
 * SubtractionOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan the Function object in which this
 * SubtractionOperator exists.
 *             @param // STOPSHIP: 6/30/2016
 *
 */
    public SubtractionOperator( int index,ArrayList<String>scan){
        super(MINUS, index, scan);
    }//end constructor











}//end class SubtractionOperator
