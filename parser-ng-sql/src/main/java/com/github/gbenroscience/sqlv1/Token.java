package com.github.gbenroscience.sqlv1;

/**
 * A single lexical token produced by {@link SqlLexer}.
 *
 * <p>
 * {@code start}/{@code end} are character offsets into the original SQL
 * source text (end-exclusive). They exist so {@link SqlParser} can capture
 * the exact raw source text of an arithmetic {@code expression} production
 * and hand it, verbatim, to ParserNG's own {@code MathExpression} compiler
 * instead of re-rendering it — see {@link SqlParser}'s class javadoc,
 * "Why expressions are captured, not rebuilt".
 *
 * @param type lexical category
 * @param text the token's own text, exactly as it appeared in the source
 * (for {@link TokenType#STRING} this is the decoded contents, without the
 * surrounding quotes)
 * @param start offset of the first character of this token in the source
 * @param end offset one past the last character of this token in the source
 *
 * @author GBEMIRO
 */
public record Token(TokenType type, String text, int start, int end) {

    /**
     * @return {@code true} if this token's type is one of {@code types}
     */
    public boolean isOneOf(TokenType... types) {
        for (TokenType t : types) {
            if (type == t) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return type + "('" + text + "' @" + start + ")";
    }
}
