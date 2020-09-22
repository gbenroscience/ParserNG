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
public final class MmultiplicationOperator extends MatrixBinaryOperator{


   
//Č

/**
 * Creates an object of class
 * MmultiplicationOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan the scanner tokens
 * MmultiplicationOperator exists.
 *
 */
    public MmultiplicationOperator(int index, ArrayList<String>scan){
        super("*", index, scan);
    }//end constructor







  

}//end class MmultiplicationOperator
