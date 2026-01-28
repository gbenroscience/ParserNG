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
public final class MSubtractionOperator extends MatrixBinaryOperator{
  
//Č

/**
 * Creates an object of class
 * MSubtractionOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan the scanner tokesn
 * MSubtractionOperator exists.
 *
 */
    public MSubtractionOperator(int index, ArrayList<String>scan){
        super("-", index, scan);
    }//end constructor











}//end class MSubtractionOperator
