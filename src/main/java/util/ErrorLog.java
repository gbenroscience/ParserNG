/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

import java.util.ArrayList;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class ErrorLog{
final ArrayList<String>logBook = new ArrayList<String>();



public void writeLog(String error){
logBook.add(error);
}
/**
 * Displays the contents of this logBook on a JTextArea object and then clears its memory.
 */
public void printLog(){
	StringBuilder builder = new StringBuilder(logBook.toString());
   
   clearLog();
}

 

public void clearLog(){
    logBook.clear();
}


}
