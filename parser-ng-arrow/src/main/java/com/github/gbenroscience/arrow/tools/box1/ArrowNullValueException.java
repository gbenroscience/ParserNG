package com.github.gbenroscience.arrow.tools.box1;

/**
 * Thrown under {@link NullPolicy#REJECT_ON_NULL} when a bound input
 * column contains a null within the evaluated row range.
 */
public class ArrowNullValueException extends RuntimeException {

    public ArrowNullValueException(String columnName, long rowIndex) {
        super("Column '" + columnName + "' contains a null value at row " + rowIndex
                + "; this ArrowBulkEvaluator is configured with NullPolicy.REJECT_ON_NULL. "
                + "Either filter/impute nulls upstream, or build the evaluator with "
                + "NullPolicy.PROPAGATE_NULL.");
    }
}
