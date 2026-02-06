/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

/**
 * Is thrown whenever an illegal matrix string is found.
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class MatrixFormatException extends Exception {

    /**
     * Creates a new instance of <code>MatrixFormatException</code> without detail message.
     */
    public MatrixFormatException() {
        super("Invalid Matrix Input");
    }


    /**
     * Constructs an instance of <code>MatrixFormatException</code> with the specified detail message.
     * @param msg the detail message.
     */
    public MatrixFormatException(String msg) {
        super(msg);
    }
}
