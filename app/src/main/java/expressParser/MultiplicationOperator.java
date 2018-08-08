/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package expressParser;

import java.util.ArrayList;
import static expressParser.Number.*;
import static expressParser.Variable.*;

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
 * @param function the Function object in which this
 * MultiplicationOperator exists.
 *
 */
    public MultiplicationOperator( int index,ArrayList<String>scan){
        super(Operator.MULTIPLY, index, scan);
    }//end constructor







  

}//end class MultiplicationOperator
