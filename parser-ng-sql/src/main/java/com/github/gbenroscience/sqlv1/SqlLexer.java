package com.github.gbenroscience.sqlv1;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Turns SQL source text into a flat list of {@link Token}s for
 * {@link SqlParser}.
 *
 * <h2>Identifiers and literals</h2>
 * An identifier starts with a letter or underscore, followed by letters,
 * digits, or underscores — the ordinary unquoted-SQL-identifier rule.
 * Quoted/bracketed identifiers ({@code "col"}, {@code [col]}, <code>`col`</code>)
 * are not supported in this v1; column and table names must be plain
 * identifiers, which also matches what ParserNG's own variable-name syntax
 * expects downstream.
 *
 * <p>
 * Numbers are unsigned {@code integer_literal | decimal_literal} per the
 * grammar — digits, optionally followed by {@code '.'} and more digits.
 * Sign is handled by the unary {@code +}/{@code -} operator in
 * {@link SqlParser}, not by the lexer. Scientific notation
 * ({@code 1e10}) is not part of the grammar and is not lexed as a single
 * number token.
 *
 * <p>
 * String literals use single or double quotes; a doubled quote character
 * ({@code ''}) inside a string of the same quote style is an escaped literal
 * quote, matching standard SQL string-literal escaping.
 *
 * <h2>Keywords</h2>
 * Keywords ({@code SELECT}, {@code FROM}, {@code WHERE}, {@code AS},
 * {@code AND}, {@code OR}, {@code NOT}, {@code BETWEEN}, {@code IN},
 * {@code IS}, {@code NULL}, {@code TRUE}, {@code FALSE}, {@code CASE},
 * {@code WHEN}, {@code THEN}, {@code ELSE}, {@code END}, {@code CAST},
 * {@code GROUP}, {@code BY}, {@code HAVING}, {@code ORDER}, {@code LIMIT},
 * {@code ASC}, {@code DESC}) are matched case-insensitively, exactly like
 * standard SQL; everything else (identifiers, i.e. column/table/function
 * names) is case-sensitive, since those names are looked up verbatim
 * against Arrow column names and ParserNG's own case-sensitive
 * function/variable names downstream.
 *
 * @author GBEMIRO
 */
public final class SqlLexer {

    private static final Map<String, TokenType> KEYWORDS = Map.ofEntries(
            Map.entry("SELECT", TokenType.SELECT),
            Map.entry("FROM", TokenType.FROM),
            Map.entry("WHERE", TokenType.WHERE),
            Map.entry("AS", TokenType.AS),
            Map.entry("AND", TokenType.AND),
            Map.entry("OR", TokenType.OR),
            Map.entry("NOT", TokenType.NOT),
            Map.entry("BETWEEN", TokenType.BETWEEN),
            Map.entry("IN", TokenType.IN),
            Map.entry("IS", TokenType.IS),
            Map.entry("NULL", TokenType.NULL),
            Map.entry("TRUE", TokenType.TRUE),
            Map.entry("FALSE", TokenType.FALSE),
            Map.entry("CASE", TokenType.CASE),
            Map.entry("WHEN", TokenType.WHEN),
            Map.entry("THEN", TokenType.THEN),
            Map.entry("ELSE", TokenType.ELSE),
            Map.entry("END", TokenType.END),
            Map.entry("CAST", TokenType.CAST),
            Map.entry("GROUP", TokenType.GROUP),
            Map.entry("BY", TokenType.BY),
            Map.entry("HAVING", TokenType.HAVING),
            Map.entry("ORDER", TokenType.ORDER),
            Map.entry("LIMIT", TokenType.LIMIT),
            Map.entry("ASC", TokenType.ASC),
            Map.entry("DESC", TokenType.DESC)
    );

    private final String src;
    private int pos;

    private SqlLexer(String src) {
        this.src = src;
    }

    /**
     * Tokenizes {@code sql} completely, including a trailing {@link Token}
     * of type {@link TokenType#EOF} so {@link SqlParser} never needs a
     * separate end-of-input check.
     *
     * @throws SqlSyntaxException on any character sequence that cannot be
     * classified as a valid token (an unterminated string, a stray
     * character such as {@code '#'} or {@code ';'}, etc.)
     */
    public static List<Token> tokenize(String sql) {
        if (sql == null) {
            throw new NullPointerException("sql must not be null");
        }
        return new SqlLexer(sql).run();
    }

    private List<Token> run() {
        List<Token> out = new ArrayList<>();
        while (true) {
            skipWhitespace();
            if (pos >= src.length()) {
                out.add(new Token(TokenType.EOF, "", pos, pos));
                return out;
            }
            out.add(nextToken());
        }
    }

    private void skipWhitespace() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            pos++;
        }
    }

    private Token nextToken() {
        int start = pos;
        char c = src.charAt(pos);

        if (Character.isLetter(c) || c == '_') {
            return identifierOrKeyword(start);
        }
        if (Character.isDigit(c)) {
            return number(start);
        }
        if (c == '\'' || c == '"') {
            return string(start, c);
        }

        switch (c) {
            case '*' -> {
                pos++;
                return new Token(TokenType.STAR, "*", start, pos);
            }
            case ',' -> {
                pos++;
                return new Token(TokenType.COMMA, ",", start, pos);
            }
            case '(' -> {
                pos++;
                return new Token(TokenType.LPAREN, "(", start, pos);
            }
            case ')' -> {
                pos++;
                return new Token(TokenType.RPAREN, ")", start, pos);
            }
            case '+' -> {
                pos++;
                return new Token(TokenType.PLUS, "+", start, pos);
            }
            case '-' -> {
                pos++;
                return new Token(TokenType.MINUS, "-", start, pos);
            }
            case '/' -> {
                pos++;
                return new Token(TokenType.SLASH, "/", start, pos);
            }
            case '%' -> {
                pos++;
                return new Token(TokenType.PERCENT, "%", start, pos);
            }
            case '^' -> {
                pos++;
                return new Token(TokenType.CARET, "^", start, pos);
            }
            case '=' -> {
                pos++;
                return new Token(TokenType.EQ, "=", start, pos);
            }
            case '!' -> {
                if (peekAt(pos + 1) == '=') {
                    pos += 2;
                    return new Token(TokenType.NEQ, "!=", start, pos);
                }
                throw new SqlSyntaxException("Unexpected character '!' (did you mean '!='?)", pos);
            }
            case '<' -> {
                if (peekAt(pos + 1) == '=') {
                    pos += 2;
                    return new Token(TokenType.LE, "<=", start, pos);
                }
                if (peekAt(pos + 1) == '>') {
                    pos += 2;
                    return new Token(TokenType.NEQ, "<>", start, pos);
                }
                pos++;
                return new Token(TokenType.LT, "<", start, pos);
            }
            case '>' -> {
                if (peekAt(pos + 1) == '=') {
                    pos += 2;
                    return new Token(TokenType.GE, ">=", start, pos);
                }
                pos++;
                return new Token(TokenType.GT, ">", start, pos);
            }
            default -> throw new SqlSyntaxException("Unexpected character '" + c + "'", pos);
        }
    }

    private char peekAt(int i) {
        return i < src.length() ? src.charAt(i) : '\0';
    }

    private Token identifierOrKeyword(int start) {
        pos++;
        while (pos < src.length() && (Character.isLetterOrDigit(src.charAt(pos)) || src.charAt(pos) == '_')) {
            pos++;
        }
        String text = src.substring(start, pos);
        TokenType kw = KEYWORDS.get(text.toUpperCase(java.util.Locale.ROOT));
        if (kw != null) {
            return new Token(kw, text, start, pos);
        }
        return new Token(TokenType.IDENTIFIER, text, start, pos);
    }

    private Token number(int start) {
        while (pos < src.length() && Character.isDigit(src.charAt(pos))) {
            pos++;
        }
        if (pos < src.length() && src.charAt(pos) == '.' && pos + 1 < src.length() && Character.isDigit(src.charAt(pos + 1))) {
            pos++;
            while (pos < src.length() && Character.isDigit(src.charAt(pos))) {
                pos++;
            }
        }
        return new Token(TokenType.NUMBER, src.substring(start, pos), start, pos);
    }

    private Token string(int start, char quote) {
        pos++; // consume opening quote
        StringBuilder sb = new StringBuilder();
        while (true) {
            if (pos >= src.length()) {
                throw new SqlSyntaxException("Unterminated string literal", start);
            }
            char c = src.charAt(pos);
            if (c == quote) {
                if (peekAt(pos + 1) == quote) {
                    sb.append(quote);
                    pos += 2;
                    continue;
                }
                pos++;
                break;
            }
            sb.append(c);
            pos++;
        }
        return new Token(TokenType.STRING, sb.toString(), start, pos);
    }
}
