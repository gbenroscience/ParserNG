package com.github.gbenroscience.sqlv1;

/**
 * Thrown by {@link SqlLexer} or {@link SqlParser} when the input text does
 * not conform to the grammar documented on {@link SqlParser}.
 *
 * <p>
 * This is a plain {@link RuntimeException} (not a checked exception) since
 * SQL text handed to {@link ArrowQuery#compile(String)} is almost always
 * either a compile-time constant or user input that the caller will surface
 * to a human directly; forcing every call site to declare/catch a checked
 * exception for a malformed string buys little. It intentionally does
 * <b>not</b> extend {@code Throwable} the way {@code ArrowExpressionEvaluator}
 * compilation failures do further downstream in parser-ng-arrow — a bad SQL
 * string is a distinct, earlier failure mode from a bad ParserNG expression
 * failing to compile.
 *
 * @author GBEMIRO
 */
public class SqlSyntaxException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    private final int position;

    public SqlSyntaxException(String message, int position) {
        super(message + " (at character " + position + ")");
        this.position = position;
    }

    public SqlSyntaxException(String message, int position, Throwable cause) {
        super(message + " (at character " + position + ")", cause);
        this.position = position;
    }

    /**
     * @return the character offset in the original SQL source text nearest
     * to where the problem was detected
     */
    public int getPosition() {
        return position;
    }
}
