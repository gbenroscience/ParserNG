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
public final class AntiLogToAnyBase extends LogOrAntiLogToAnyBase {

    //Č
    /**
     *
     * @param index the index of this object in its parent Function object's
     * scanner.
     * @param scan the Function object in which this RemainderOperator exists.
     */
    public AntiLogToAnyBase(int index, ArrayList<String> scan) {
        super("log-¹", index, scan);
    }//end constructor
}
