/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import java.io.Serializable;

/**
 * Defines the allowed return types
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public enum TYPE implements Serializable{
    /**
     * MATRIX is a 2D store of numbers
     * VECTOR is a 1D store of numbers
     * ARRAY is a 1D store of numbers and text
     * STRING is textual data
     * VOID is no data
     * ALGEBRAIC_EXPRESSION is textual data that is math and may be given properties in the future(may map to MathExpression)
     * ERROR is an error type
     * BOOLEAN is logic
     */
    MATRIX, VECTOR, ARRAY, NUMBER, STRING, VOID, ALGEBRAIC_EXPRESSION, ERROR, BOOLEAN
}
