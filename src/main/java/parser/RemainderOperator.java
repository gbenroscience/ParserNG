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
public final class RemainderOperator extends BinaryOperator{


   
//Č

/**
 * Creates an object of class
 * RemainderOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan the Function object in which this
 * RemainderOperator exists.
 *
 */
    public RemainderOperator( int index,ArrayList<String>scan){
        super(REMAINDER, index, scan);
    }//end constructor







    


}//end class RemainderOperator
