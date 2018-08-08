/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import expressParser.*;
import java.util.ArrayList;

/**
 *
 * @author GBEMIRO
 */
public final class SubtractionOperator extends MatrixBinaryOperator{
  
//Č

/**
 * Creates an object of class
 * SubtractionOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param function the Function object in which this
 * SubtractionOperator exists.
 *
 */
    public SubtractionOperator( int index,ArrayList<String>scan){
        super("-", index, scan);
    }//end constructor











}//end class SubtractionOperator
