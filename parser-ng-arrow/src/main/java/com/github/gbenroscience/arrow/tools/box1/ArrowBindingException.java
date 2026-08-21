package com.github.gbenroscience.arrow.tools.box1;

/**
 * Thrown when binding ParserNG variable names to Arrow schema columns
 * cannot be satisfied — a missing column, a wrong declared variable
 * count, a failed expression compile, etc.
 */
public class ArrowBindingException extends RuntimeException {

    public ArrowBindingException(String message) {
        super(message);
    }

    public ArrowBindingException(String message, Throwable cause) {
        super(message, cause);
    }
}
