package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.old;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal lexer producing the same flat token vocabulary shown in ParserNG's
 * scanned form: single-character tokens for '(', ')', '[', ']', ',', '+',
 * '-', '*', '/', '^', plus multi-character NUMBER and IDENT tokens. This is
 * not a reimplementation of ParserNG's real scanner — it exists so
 * CoefficientExtractor can accept a raw expression string directly, in
 * addition to an already-scanned token list, and treat both the same way
 * past this point.
 *
 * <h2>Scope</h2>
 * Handles exactly the vocabulary needed for arithmetic expressions with
 * function calls and indexed variable access: numbers (integer or decimal,
 * no exponent notation), identifiers (letter followed by letters/digits/
 * underscore), and the operators/punctuation above. Whitespace is skipped.
 * Anything else (an unrecognized character) throws IllegalArgumentException
 * naming the offending character and its position, rather than silently
 * skipping it.
 */
public final class Lexer {

    private Lexer() {
    }

    public static List<String> tokenize(String expression) {
        if (expression == null) {
            throw new IllegalArgumentException("expression must not be null");
        }

        List<String> tokens = new ArrayList<>();
        int i = 0;
        int n = expression.length();

        while (i < n) {
            char c = expression.charAt(i);

            if (Character.isWhitespace(c)) {
                i++;
                continue;
            }

            if (c == '(' || c == ')' || c == '[' || c == ']' || c == ','
                    || c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
                tokens.add(String.valueOf(c));
                i++;
                continue;
            }

            if (Character.isDigit(c)) {
                int start = i;
                while (i < n && Character.isDigit(expression.charAt(i))) {
                    i++;
                }
                if (i < n && expression.charAt(i) == '.') {
                    i++;
                    while (i < n && Character.isDigit(expression.charAt(i))) {
                        i++;
                    }
                }
                tokens.add(expression.substring(start, i));
                continue;
            }

            if (Character.isLetter(c) || c == '_') {
                int start = i;
                while (i < n && (Character.isLetterOrDigit(expression.charAt(i)) || expression.charAt(i) == '_')) {
                    i++;
                }
                tokens.add(expression.substring(start, i));
                continue;
            }

            throw new IllegalArgumentException(
                    "Unrecognized character '" + c + "' at position " + i + " in expression: " + expression);
        }

        return tokens;
    }
}