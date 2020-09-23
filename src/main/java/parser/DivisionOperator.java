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
public final class DivisionOperator extends BinaryOperator {

    /**
     *
     * @param index the index of this object in its parent Function object's
     * scanner.
     * @param scan The scanner output
     */
    public DivisionOperator(int index, ArrayList<String> scan) {
        super(DIVIDE, index, scan);
    }//end constructor

}//end class DivisionOperator
