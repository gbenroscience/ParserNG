/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package com.itis.libs.parserng.android.expressParser;

import java.util.ArrayList;


/**
 *
 * @author GBEMIRO
 */
public final class DivisionOperator extends BinaryOperator{



//Č

/**
 * Creates an object of class
 * DivisionOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param function the Function object in which this
 * DivisionOperator exists.
 *
 */
    public DivisionOperator( int index,ArrayList<String>scan){
        super(DIVIDE, index,  scan);
    }//end constructor


    

}//end class DivisionOperator
