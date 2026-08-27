package com.github.gbenroscience.arrow.tools.box;

/**
 * Thrown when an {@link ArrowBulkEvaluator} cannot bind a compiled
 * expression's required variables to the Arrow columns supplied at
 * evaluation time — e.g. a missing column, a row-count mismatch between
 * bound columns, an output vector that has not been sized, or a column
 * whose Arrow type is not directly compatible with zero-copy double
 * evaluation.
 */
public class ArrowBindingException extends RuntimeException {

    public ArrowBindingException(String message) {
        super(message);
    }

    public ArrowBindingException(String message, Throwable cause) {
        super(message, cause);
    }
}