/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import java.io.Serializable;

/**
 * Defines the allowed return types
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public enum TYPE implements Serializable{

    MATRIX, LIST, NUMBER, STRING, VOID, ALGEBRAIC_EXPRESSION, ERROR

}
