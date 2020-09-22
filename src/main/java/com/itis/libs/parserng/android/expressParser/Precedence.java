/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package com.itis.libs.parserng.android.expressParser;

/**
 *
 * @author GBEMIRO
 */
public class Precedence {
    private int value;

    public Precedence(int value) {
     setValue(value);
    }

    public void setValue(int value) {
       this.value = (value>=0)?value:0;
    }

    public int getValue(){
        return value;
    }

    

    @Override
    public String toString() {
        return " Precedence = "+value;
    }



}
