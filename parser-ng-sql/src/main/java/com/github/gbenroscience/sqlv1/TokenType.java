package com.github.gbenroscience.sqlv1;

/**
 * Lexical categories produced by {@link SqlLexer}.
 *
 * <p>
 * This is deliberately a small, closed set: parser-ng-sql speaks exactly the
 * grammar documented on {@link SqlParser} (a {@code SELECT ... FROM ...
 * [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] [LIMIT ...]} shape
 * over ParserNG arithmetic expressions, plus {@code CASE}/{@code CAST} as
 * expression-level constructs) and nothing more. There is no token type for
 * {@code JOIN}, subqueries, etc. — those are out of scope by design; see
 * the parser-ng-sql module javadoc.
 *
 * @author GBEMIRO
 */
public enum TokenType {

    // --- keywords (case-insensitive in source text) ---
    SELECT,
    FROM,
    WHERE,
    AS,
    AND,
    OR,
    NOT,
    BETWEEN,
    IN,
    IS,
    NULL,
    TRUE,
    FALSE,
    CASE,
    WHEN,
    THEN,
    ELSE,
    END,
    CAST,
    GROUP,
    BY,
    HAVING,
    ORDER,
    LIMIT,
    ASC,
    DESC,

    // --- literals / names ---
    IDENTIFIER,
    NUMBER,
    STRING,

    // --- punctuation ---
    STAR,       // '*' -- SELECT * or multiplication, disambiguated by the parser
    COMMA,
    LPAREN,
    RPAREN,

    // --- arithmetic operators (ParserNG's own `expression` grammar) ---
    PLUS,
    MINUS,
    SLASH,
    PERCENT,
    CARET,

    // --- comparison operators ---
    EQ,         // '='
    NEQ,        // '!=' or '<>'
    LT,
    LE,
    GT,
    GE,

    EOF
}
