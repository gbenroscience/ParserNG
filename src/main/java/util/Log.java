/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

/**
 *
 * @author GBEMIRO JIBOYE , gbenroscience@gmail.com
 */
public class Log {

    public static final void e(String tag ,String msg) {
           System.err.printf("%s    , %s",tag, msg);
    }

    public static final void d(String tag ,String msg) {
           System.out.printf("%s    , %s",tag, msg);
    }

    public static final void i(String tag ,String msg) {
        System.out.printf("%s    , %s",tag, msg);
    }

}
