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
public final class MultiplicationOperator extends BinaryOperator{


   
//Č

/**
 * Creates an object of class
 * MultiplicationOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan The scanner output
 *
 */
    public MultiplicationOperator( int index,ArrayList<String>scan){
        super(MULTIPLY, index, scan);
    }//end constructor







  

}//end class MultiplicationOperator
