package com.github.gbenroscience.sqlv1;

/**
 * Lexical categories produced by {@link SqlLexer}.
 *
 * <p>
 * parser-ng-sql speaks the {@code SELECT ... FROM ... [WHERE ...] [GROUP BY
 * ...] [HAVING ...] [ORDER BY ...] [LIMIT ...]} shape over ParserNG
 * arithmetic expressions documented on {@link SqlParser}, plus {@code CASE}/
 * {@code WHEN}/{@code THEN}/{@code ELSE}/{@code END} and {@code CAST} as
 * expression-level constructs. There is still no token type for
 * {@code JOIN}, subqueries, {@code INSERT}/{@code UPDATE}/{@code DELETE},
 * transactions, or catalogs — those remain out of scope by design; see the
 * parser-ng-sql module javadoc.
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
    ASC,
    DESC,
    LIMIT,

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