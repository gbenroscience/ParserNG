package com.github.gbenroscience.sqlv1;

/**
 * Thrown by {@link ArrowQuery#execute(org.apache.arrow.vector.VectorSchemaRoot)}
 * when compiling the query's {@code SELECT}/{@code WHERE} expressions against
 * a concrete {@code VectorSchemaRoot} fails.
 *
 * <p>
 * {@code ArrowExpressionEvaluators.compile}/{@code compileF32} in
 * parser-ng-arrow declare {@code throws Throwable} (a compiled ParserNG
 * expression can, in principle, fail for almost any reason — a bad function
 * name, a GPU backend that fails to bootstrap, and so on). {@link ArrowQuery}
 * wraps whatever it catches from those calls in this unchecked exception
 * rather than propagating a raw {@code Throwable}, so ordinary calling code
 * does not need to declare or catch {@code Throwable} just to call
 * {@link ArrowQuery#execute}. The original failure is always available via
 * {@link #getCause()}.
 *
 * <p>
 * This does <b>not</b> wrap {@link SqlSyntaxException} — a malformed SQL
 * string fails earlier, during {@link ArrowQuery#compile(String)}, and
 * {@link SqlSyntaxException} is thrown directly from there since it is
 * already unchecked and needs no wrapping.
 *
 * @author GBEMIRO
 */
public class ArrowSqlException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public ArrowSqlException(String message, Throwable cause) {
        super(message, cause);
    }
}
