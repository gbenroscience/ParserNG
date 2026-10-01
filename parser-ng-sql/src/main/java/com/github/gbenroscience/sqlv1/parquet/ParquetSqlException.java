package com.github.gbenroscience.sqlv1.parquet;

/**
 * Thrown by the Parquet bridge for problems that are about the <em>data or the
 * file</em> rather than about SQL syntax: an unsupported column type, a value
 * that cannot be represented without loss, a path outside the allowed root.
 * Messages always name the column, its type and (where relevant) the file, so
 * nothing is ever silently mis-decoded or coerced.
 */
public class ParquetSqlException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public ParquetSqlException(String message) {
        super(message);
    }

    public ParquetSqlException(String message, Throwable cause) {
        super(message, cause);
    }
}
