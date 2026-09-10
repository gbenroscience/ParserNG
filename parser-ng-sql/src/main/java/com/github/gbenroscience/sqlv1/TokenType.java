package com.github.gbenroscience.sqlv1;

/**
 * Lexical categories produced by {@link SqlLexer}.
 *
 * <p>
 * This is deliberately a small, closed set: parser-ng-sql speaks exactly the
 * grammar documented on {@link SqlParser} (a {@code SELECT ... FROM ...
 * [WHERE ...]} shape over ParserNG arithmetic expressions) and nothing more.
 * There is no token type for {@code JOIN}, {@code GROUP BY},
 * {@code ORDER BY}, subqueries, etc. — those are out of scope by design; see
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
