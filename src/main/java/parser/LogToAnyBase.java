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
public final class LogToAnyBase extends LogOrAntiLogToAnyBase{
/**
 * Creates an object of class
 * LogTOperator
 *
 * @param index the index of this object in its
 * parent Function object's scanner.
 * @param scan The scanner output
 *
 */
    public LogToAnyBase( int index,ArrayList<String>scan){
        super("log", index, scan);
    }//end constructor


}
