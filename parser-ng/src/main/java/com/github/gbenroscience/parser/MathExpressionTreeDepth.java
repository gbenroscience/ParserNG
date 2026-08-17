/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser;

import com.github.gbenroscience.interfaces.Savable;

import java.util.regex.Pattern;

/**
 *
 * @author GBEMIRO
 */
/**
 * Production-hardened class to compute the depth (height) of the abstract
 * syntax tree (AST) for a mathematical / logical expression, used to
 * pre-size a stack-based evaluator before it runs.
 *
 * Features: - Handles numbers (integers, decimals, scientific notation like
 * 1.2e-3) - Variables (e.g., x, varName_123) - String literals, single- or
 * double-quoted, e.g. 'hello' or "hello" (see {@link #consumeStringLiteral})
 * - Binary operators: + - * / ^
 * (power, right-associative) - Unary + - ! √ (square root) - Prefix "nth
 * root" notation: a run of superscript digits immediately before '√', e.g.
 * "³√9" (cube root of 9) - Postfix Unicode superscript exponents, e.g.
 * "x²", "3³", "y²³" (== y^23) - Binary combinatoric operators nPr and nCr,
 * written as 'Р' (Cyrillic Er, U+0420) and 'Č' (U+010C) respectively, e.g.
 * "9Р3" and "6Č5" — same precedence tier as '*' and '/' - Functions with any number of arguments
 * (e.g., sin(x), max(a, b, c+ d)) - Parentheses for grouping AND for
 * comma-separated literal lists, e.g. matrix data "(3,1,4,7)" - Implicit
 * multiplication where a number, ')', identifier, '(', '@', '√', or a
 * string-literal quote is juxtaposed directly against a following
 * identifier, '(', '@', '√', or quote
 * with no explicit operator between them, e.g. "3t" == "3*t", "5sin(4)"
 * == "5*sin(4)", "3(x+1)" == "3*(x+1)", "(x+1)(x-1)" == "(x+1)*(x-1)",
 * "3'x'" == "3*'x'" — matching ParserNG's own lenient grammar downstream,
 * at the same
 * precedence tier as an explicit '*'; deliberately NOT extended to two
 * bare numbers ("1 2" is still rejected, not read as "1*2") or to any
 * identifier that could be read as the OR/AND keyword ("3ORx" is still
 * rejected, not read as "3*ORx") - Derivative
 * index notation "name[n]" (n a non-negative integer literal), e.g. "y[3]"
 * for y'''(x) — recognized ONLY directly inside a call to one of the four
 * differential-equation functions diffeqn / diffeqnPath / diffeqnHO /
 * diffeqnPathHO; '[' is not a general-purpose grouping delimiter and is a
 * parse error anywhere else - Relational operators: &gt; &lt; &gt;= &lt;= == != - Logical
 * operators: &amp;&amp; || (and the symbolic single &amp; / |), plus the
 * textual keywords OR / AND (word-boundary safe, so identifiers like
 * "orange" or "android" are never mistaken for the keyword) - Anonymous
 * function and matrix definitions via '@', e.g. "@(x)sin(x)",
 * "@(x,y,z)=2*x+3*y+4*z", "@(2,2)(3,1,4,7)" - No external libraries,
 * single-pass O(n) parsing - Spaces are ignored (except inside string
 * literals, where they are preserved as ordinary content)
 *
 * Tree depth definition: - Leaf (number, variable, or string literal) = 1 -
 * Binary operator
 * node = 1 + max(left depth, right depth) - Unary operator node (prefix
 * +/-/!/√, or a postfix Unicode superscript exponent run) = 1 + operand
 * depth - Function node (named call, parenthesized comma-list, or anonymous
 * '@' function/matrix) = max over items of (depth of that item + number of
 * items already evaluated and left on the stack), never less than 1 -
 * Parentheses around a single item do not add extra depth (pure grouping)
 *
 * Example: "2 + 3 * 4" -> depth 3 ((2 + (3 * 4))) "-2^3" -> depth 3 (- (2 ^ 3))
 * "2^-3" -> depth 3 (2 ^ (-3)) "sin(2 + 3 * 4)" -> depth 4 "(1 + (2 + (3 +
 * 4)))"-> depth 4 "2²+3³+√9" -> depth 4 ((2²) + ((3³) + (√9)))
 *
 * <h2>Production / correctness guarantees</h2>
 * <ul>
 *   <li>Any malformed, incomplete, or unsupported input throws a
 *       {@link ParseException} rather than silently returning an incomplete
 *       or wrong depth. A "successful" {@link #calculate()} call is a
 *       guarantee that the <em>entire</em> input string was consumed and
 *       is structurally well formed.</li>
 *   <li>Pathologically deep nesting (parentheses, function calls, or long
 *       chains of unary signs) fails fast with a {@link ParseException}
 *       instead of risking a JVM {@link StackOverflowError} that could take
 *       down the whole batch/bulk evaluation run. As a last line of
 *       defense, any {@link StackOverflowError} that does slip through is
 *       caught and converted into a {@link ParseException} as well.</li>
 *   <li>The four differential-equation functions {@code diffeqn},
 *       {@code diffeqnPath}, {@code diffeqnHO}, and {@code diffeqnPathHO}
 *       may each ONLY appear as the root of the entire input expression —
 *       i.e. the whole (trimmed) input must be exactly one call to one of
 *       these functions, optionally wrapped in any number of pure grouping
 *       parentheses, with nothing else before or after it. This is enforced
 *       structurally while parsing (not as a post-hoc string scan), so it
 *       composes correctly with every other feature above.
 *       {@code diffeqn(...)} is legal, and so is {@code (((diffeqn(...))))}
 *       (the RPN tokenizer downstream discards pure grouping parens anyway);
 *       {@code 2+diffeqn(...)}, {@code diffeqn(...)-2},
 *       {@code sin(diffeqn(...))}, {@code diffeqn(...)*diffeqn(...)}, and
 *       {@code (diffeqn(...), 2)} (a two-item list, not a pure wrap) are all
 *       rejected with a {@link ParseException}. (Assignment stripping, e.g.
 *       {@code "A = diffeqn(...)"} -> {@code "diffeqn(...)"}, is expected
 *       to have already been handled by a preprocessor before the string
 *       reaches this class.)</li>
 * </ul>
 *
 * <h2>Thread-safety / reuse</h2>
 * This class is <b>not thread-safe</b> and is intended for single use: create
 * one instance per expression, call {@link #calculate()} once, and discard
 * it. A second call to {@link #calculate()} on the same instance throws
 * {@link IllegalStateException}.
 */
public class MathExpressionTreeDepth implements Savable {

    private static final long serialVersionUID = 1L;

    /**
     * Upper bound on structural nesting (parentheses, function calls, and
     * chained unary signs). Prevents pathological input from blowing the
     * JVM call stack; tune to taste for your environment.
     */
    private static final int MAX_NESTING_DEPTH = 256;

    /**
     * Strict grammar for a numeric literal: digits, optional fractional
     * part, optional scientific-notation exponent. Rejects malformed
     * tokens like "5.", "1e", or "1.2.3" (the trailing ".3" would simply be
     * left unconsumed and caught by the trailing-content check anyway, but
     * failing here gives a much more precise error).
     */
    private static final Pattern NUMBER_PATTERN =
            Pattern.compile("(\\d+(\\.\\d+)?|\\.\\d+)([eE][+-]?\\d+)?");

    /**
     * Prefix unary "square root" operator, e.g. "√9", "√√9", "-√9". Treated
     * exactly like the other unary prefix operators (+ - !): it recurses via
     * {@link #parseUnary()} so chains and combinations with signs work, and
     * contributes {@code 1 + operandDepth} like any unary node.
     */
    private static final char ROOT_CHAR = '\u221A'; // '√'

    /**
     * Double-quote string literal delimiter, e.g. {@code "hello"}.
     *
     * @see #consumeStringLiteral(char)
     */
    private static final char DOUBLE_QUOTE = '"';

    /**
     * Single-quote string literal delimiter, e.g. {@code 'hello'}.
     *
     * @see #consumeStringLiteral(char)
     */
    private static final char SINGLE_QUOTE = '\'';

    /**
     * Binary "permutation" operator, e.g. {@code 9Р3} (nPr — the number of
     * ways to arrange 3 items out of 9). This is the Cyrillic capital letter
     * Er ('Р', U+0420) — visually similar to the Latin 'P' but a distinct
     * code point — chosen so it never collides with an ordinary identifier
     * character. Because {@link Character#isLetter(char)} is {@code true}
     * for this character, it is explicitly excluded from identifier
     * consumption (see {@link #isReservedOperatorChar(char)} and
     * {@link #consumeIdentifier()}) so it is always tokenized as this
     * operator, never folded into a variable name.
     */
    private static final char PERMUTATION_CHAR = '\u0420'; // 'Р'

    /**
     * Binary "combination" operator, e.g. {@code 6Č5} (nCr — the number of
     * ways to choose 5 items out of 6, order not mattering). This is the
     * Latin capital letter C with caron ('Č', U+010C). Like
     * {@link #PERMUTATION_CHAR}, it is a letter as far as
     * {@link Character#isLetter(char)} is concerned, so it too is excluded
     * from identifier consumption to guarantee it is always tokenized as
     * this operator.
     */
    private static final char COMBINATION_CHAR = '\u010C'; // 'Č'

    /**
     * All ten Unicode superscript digit characters (⁰¹²³⁴⁵⁶⁷⁸⁹), in the same
     * order as their plain-ASCII counterparts '0'-'9'. A run of one or more
     * of these immediately following a parsed primary/power base is treated
     * as a single postfix "raised to the power of &lt;these digits&gt;"
     * operator, e.g. "x²" == "x^2" and "y²³" == "y^23" (the whole run forms
     * one literal exponent, not a chain of separate '^' applications).
     */
    private static final String SUPERSCRIPT_DIGITS = "\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079";

    private final String expr;
    private int pos;

    // Counters
    private int binaryOpCount = 0;
    private int divOpCount = 0;
    private int unaryOpCount = 0;
    private int functionCount = 0;

    // Guards against pathological recursion (nested parens/functions, long
    // unary chains).
    private int nestingDepth = 0;

    // Counts how many diffeqn-family function calls (diffeqn, diffeqnPath,
    // diffeqnHO, diffeqnPathHO) we are currently lexically inside of. While
    // this is > 0, "name[n]" derivative-index notation (e.g. "y[3]") is
    // recognized on a bare variable in parsePrimary(); everywhere else '['
    // is just an unrecognized character. A counter (rather than a boolean)
    // so that a diffeqn call nested inside another one still counts as "inside".
    // NOTE: a diffeqn call nested inside another diffeqn-family call is now
    // itself rejected by the root-only rule before this could ever exceed 1
    // in practice, but the counter is kept as-is since it does no harm and
    // keeps this field's original meaning intact.
    private int diffEqDepth = 0;

    // Guards against reusing a single instance across multiple parses.
    private boolean consumed = false;

    /**
     * NOTE: whitespace is intentionally NOT stripped up front (the original
     * implementation did {@code replaceAll("\\s+", "")} in the constructor).
     * Doing so destroys token boundaries: "a AND b" and "aANDb" become the
     * same string, so a textual keyword operator can never be reliably told
     * apart from an identifier that merely starts with the same letters
     * (e.g. "orange", "android"). Whitespace is instead skipped lazily, at
     * each point where the parser is deciding what token comes next (see
     * {@link #skipWhitespace()}), while the low-level token consumers
     * ({@link #consumeNumber()}, {@link #consumeIdentifier()}, {@link
     * #consumeStringLiteral(char)}) deliberately
     * use the raw, non-skipping {@link #peek()}/{@link #nextChar()} so that
     * whitespace still correctly terminates a number or identifier, and is
     * preserved verbatim as content inside a string literal.
     */
    public MathExpressionTreeDepth(String expression) {
        this.expr = expression == null ? "" : expression;
        this.pos = 0;
    }

    /**
     * Thrown for any structurally invalid, incomplete, or unsupported
     * expression: unknown characters, unclosed parentheses/functions,
     * malformed numeric literals, unterminated string literals, trailing
     * unparsed content, or excessive
     * nesting. Unchecked, so a bulk evaluator can catch it per-expression
     * without forcing a checked-exception signature everywhere.
     */
    public static class ParseException extends RuntimeException {

        private static final long serialVersionUID = 1L;
        private final int position;

        public ParseException(String message, int position) {
            super(message + " [position " + position + "]");
            this.position = position;
        }

        public ParseException(String message, int position, Throwable cause) {
            super(message + " [position " + position + "]", cause);
            this.position = position;
        }

        /** Index into the original expression string where parsing failed. */
        public int getPosition() {
            return position;
        }
    }

    public static class Result implements Savable {

        private static final long serialVersionUID = 1L;
        public final int depth;
        public final int binaryOperators;
        public final int unaryOperators;
        public final int divOperators;
        public final int functions;

        public Result(int depth, int bin, int div, int un, int funcs) {
            this.depth = depth;
            this.binaryOperators = bin;
            this.divOperators = div;
            this.unaryOperators = un;
            this.functions = funcs;
        }

        @Override
        public String toString() {
            return String.format("depth: %d | binary ops: %d | unary ops: %d | functions: %d",
                    depth, binaryOperators, unaryOperators, functions);
        }
    }

    /**
     * Parses the expression and computes its evaluation-stack depth.
     *
     * @return the computed {@link Result}
     * @throws IllegalStateException if called more than once on this instance
     * @throws ParseException        if the expression is malformed, incomplete,
     *                                unsupported, or nested too deeply
     */
    public Result calculate() {
        if (consumed) {
            throw new IllegalStateException(
                    "calculate() was already invoked on this instance. "
                    + "Create a new MathExpressionTreeDepth per expression.");
        }
        consumed = true;

        if (expr.trim().isEmpty()) {
            return new Result(0, 0, 0, 0, 0);
        }

        try {
            int depth = parseExpression();
            skipWhitespace();
            if (pos != expr.length()) {
                throw new ParseException(
                        "Unexpected trailing content starting with '" + peek() + "'", pos);
            }
            return new Result(depth, binaryOpCount, divOpCount, unaryOpCount, functionCount);
        } catch (StackOverflowError overflow) {
            // Belt-and-suspenders: the nesting guard should catch this first,
            // but never let a raw Error escape into a bulk evaluator.
            throw new ParseException("Expression is too deeply nested to parse safely", pos, overflow);
        }
    }

    private void enterNesting() {
        nestingDepth++;
        if (nestingDepth > MAX_NESTING_DEPTH) {
            throw new ParseException(
                    "Expression nesting exceeds maximum of " + MAX_NESTING_DEPTH, pos);
        }
    }

    private void exitNesting() {
        nestingDepth--;
    }

    /**
     * Advances {@link #pos} past any run of whitespace. Called only at
     * "decision points" (where the parser is about to look at {@link #peek()}
     * to decide what comes next) — never from inside {@link #consumeNumber()},
     * {@link #consumeIdentifier()}, or {@link #consumeStringLiteral(char)},
     * which rely on whitespace to correctly terminate (or, for a string
     * literal, to be preserved as ordinary content within) the token
     * they're consuming.
     */
    private void skipWhitespace() {
        while (pos < expr.length() && Character.isWhitespace(expr.charAt(pos))) {
            pos++;
        }
    }

    /**
     * True if {@code c} is one of the ten Unicode superscript digit
     * characters (⁰¹²³⁴⁵⁶⁷⁸⁹).
     */
    private static boolean isSuperscriptDigit(char c) {
        return SUPERSCRIPT_DIGITS.indexOf(c) >= 0;
    }

    /**
     * True for a character that is reserved as an operator token despite
     * {@link Character#isLetter(char)} being {@code true} for it —
     * currently {@link #PERMUTATION_CHAR} and {@link #COMBINATION_CHAR}.
     * Both {@link #consumeIdentifier()} and the identifier branch of
     * {@link #parsePrimary()} check this so these characters are always
     * tokenized as operators and never absorbed into a variable/function
     * name.
     */
    private static boolean isReservedOperatorChar(char c) {
        return c == PERMUTATION_CHAR || c == COMBINATION_CHAR;
    }

    /**
     * True if {@code c} begins a new primary/unary term reachable with no
     * explicit operator in between — the trigger for ParserNG's implicit-
     * multiplication leniency, e.g. {@code "3t"} == {@code "3*t"},
     * {@code "5sin(4)"} == {@code "5*sin(4)"}, and {@code "3'x'"} ==
     * {@code "3*'x'"}. Applied only from {@link
     * #parseMultiplicative()}, at exactly the same precedence tier as an
     * explicit {@code *}, so it composes correctly with every other
     * operator (e.g. {@code "2+3t"} is {@code 2+(3*t)}, not
     * {@code (2+3)*t}, matching standard math convention).
     *
     * <p>Deliberately conservative in two ways:</p>
     * <ul>
     *   <li>A bare digit/{@code '.'} is NOT a trigger, so {@code "1 2"} is
     *       still rejected as a likely missing operator — exactly as this
     *       class's own bundled tests document — rather than silently read
     *       as {@code "1*2"}.</li>
     *   <li>An identifier that could be read as the textual {@code OR}/
     *       {@code AND} keyword is excluded via {@link #looksLikeOrAndPrefix()},
     *       so {@code "3ORx"} remains rejected as "ambiguous run-together
     *       text" exactly as before, rather than newly guessed at as
     *       {@code "3*ORx"}.</li>
     * </ul>
     *
     * <p>A string literal's opening quote ({@code '} or {@code "}) is also a
     * trigger, on the same footing as {@code '('} or {@code '@'}: a string
     * is just another primary value, so {@code 3'x'}, {@code (x+1)"y"}, and
     * {@code 'a'x} are all read as implicit multiplication exactly like
     * their numeric/identifier counterparts.</p>
     */
    private boolean isImplicitMultiplicationTrigger(char c) {
        if (c == '(' || c == '@' || c == ROOT_CHAR || c == SINGLE_QUOTE || c == DOUBLE_QUOTE) {
            return true;
        }
        if (Character.isLetter(c) && !isReservedOperatorChar(c)) {
            return !looksLikeOrAndPrefix();
        }
        return false;
    }

    /**
     * True if the next up to three characters, case-insensitively, could be
     * read as the start of the {@code OR}/{@code AND} textual keyword —
     * regardless of whether a clean word boundary follows. Deliberately
     * broader than {@link #matchesWord(String)} (which requires a real word
     * boundary): this is a leniency-suppression guard, not a keyword match,
     * so it errs toward rejecting the ambiguous case (an identifier like
     * "Orbit" or "Andrew" directly after a number with no operator, e.g.
     * "3Orbit") rather than guessing — consistent with this class's
     * existing "reject rather than guess" philosophy for run-together text.
     */
    private boolean looksLikeOrAndPrefix() {
        int remaining = expr.length() - pos;
        String upcoming = expr.substring(pos, pos + Math.min(3, remaining)).toUpperCase();
        return upcoming.startsWith("OR") || upcoming.startsWith("AND");
    }

    /**
     * The four differential-equation function names for which the
     * {@code name[n]} derivative-index notation (see {@link #diffEqDepth})
     * is recognized inside their argument list, and which are subject to
     * the root-only rule enforced in {@link #parsePrimary()}: a call to one
     * of these functions may only appear as the entire (trimmed) input
     * expression — optionally wrapped in any number of pure grouping
     * parentheses — never nested inside another expression, never combined
     * with an operator, and never with anything else following it.
     */
    private static final java.util.Set<String> DIFF_EQ_FUNCTIONS = new java.util.HashSet<>(
            java.util.Arrays.asList("diffeqn", "diffeqnPath", "diffeqnHO", "diffeqnPathHO"));

    /**
     * True if every character in {@code expr[from, to)} is either
     * whitespace or {@code allowed}. Used by the diffeqn-family root-only
     * check to confirm that everything surrounding a candidate root call is
     * nothing but pure grouping parentheses (and whitespace) — the only
     * thing standing before an identifier at the very start of the
     * (sub)expression can grammatically be is a chain of grouping '(' opens
     * (a function call needs a preceding identifier; a list needs a comma
     * inside), so this check is sufficient to distinguish
     * "(((diffeqn(...))))" (allowed) from "sin(diffeqn(...))" or
     * "(diffeqn(...), 2)" (rejected, since 's'/'i'/'n' or ',' would appear
     * in that span).
     */
    private boolean isOnlyWhitespaceAndChar(int from, int to, char allowed) {
        for (int i = from; i < to; i++) {
            char c = expr.charAt(i);
            if (c != allowed && !Character.isWhitespace(c)) {
                return false;
            }
        }
        return true;
    }

    private int parseRelationalLogical() {
        int maxDepth = parseAdditive();

        while (true) {
            skipWhitespace();
            char c = peek();

            // 1. Handle >, <, >=, <=, ==, !=
            if (c == '>' || c == '<' || c == '=' || c == '!') {
                nextChar(); // Consume the primary symbol
                if (peek() == '=') {
                    nextChar(); // Consume the '=' to form >=, <=, ==, !=
                }
                binaryOpCount++;
                int rightDepth = parseAdditive();
                // 1 slot for the left side's result, plus the right side's requirement
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            }
            // 2. Handle && (Logical AND, symbolic)
            else if (c == '&') {
                nextChar();
                if (peek() == '&') {
                    nextChar(); // Consume the second '&'
                }
                binaryOpCount++;
                int rightDepth = parseAdditive();
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            }
            // 3. Handle || (Logical OR, symbolic)
            else if (c == '|') {
                nextChar();
                if (peek() == '|') {
                    nextChar(); // Consume the second '|'
                }
                binaryOpCount++;
                int rightDepth = parseAdditive();
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            }
            // 4. Textual "OR" / "AND" keywords. Word-boundary checked so that
            //    identifiers like "orange" or "android" are never mistaken
            //    for the keyword.
            else if ((c == 'O' || c == 'o') && matchesWord("OR")) {
                consumeWord(2);
                binaryOpCount++;
                int rightDepth = parseAdditive();
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            } else if ((c == 'A' || c == 'a') && matchesWord("AND")) {
                consumeWord(3);
                binaryOpCount++;
                int rightDepth = parseAdditive();
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            } else {
                break;
            }
        }
        return maxDepth;
    }

    /**
     * Case-insensitively checks whether {@code word} begins at the current
     * position AND is followed by a non-identifier character (or end of
     * input) — i.e. it is not merely the prefix of a longer identifier.
     */
    private boolean matchesWord(String word) {
        int len = word.length();
        if (pos + len > expr.length()) {
            return false;
        }
        for (int i = 0; i < len; i++) {
            if (Character.toUpperCase(expr.charAt(pos + i)) != word.charAt(i)) {
                return false;
            }
        }
        if (pos + len < expr.length()) {
            char next = expr.charAt(pos + len);
            if (Character.isLetterOrDigit(next) || next == '_') {
                return false;
            }
        }
        return true;
    }

    private void consumeWord(int len) {
        pos += len;
    }

    // ──────────────────────────────────────────────
    //  Parser levels
    // ──────────────────────────────────────────────
    private int parseExpression() {
        enterNesting();
        try {
            return parseRelationalLogical();
        } finally {
            exitNesting();
        }
    }

    private int parseAdditive() {
        int maxDepth = parseMultiplicative();
        while (true) {
            skipWhitespace();
            char c = peek();
            if (c == '+' || c == '-') {
                nextChar();
                binaryOpCount++;
                int rightDepth = parseMultiplicative();
                // We need 1 slot for the left side's accumulated result, plus whatever the right side needs
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            } else {
                break;
            }
        }
        return maxDepth;
    }

    /**
     * Multiplicative-precedence tier: {@code *}, {@code /}, and the two
     * combinatoric binary operators {@link #PERMUTATION_CHAR} ({@code Р},
     * nPr) and {@link #COMBINATION_CHAR} ({@code Č}, nCr) all share this
     * level, e.g. {@code 9Р3} and {@code 6Č5} bind exactly like {@code *}
     * and {@code /} do — tighter than {@code +}/{@code -}, looser than
     * unary/power.
     */
    private int parseMultiplicative() {
        int maxDepth = parseUnary();
        while (true) {
            skipWhitespace();
            char c = peek();
            if (c == '*' || c == '/' || c == PERMUTATION_CHAR || c == COMBINATION_CHAR) {
                nextChar();
                binaryOpCount++;
                if (c == '/') {
                    divOpCount++;
                }
                int rightDepth = parseUnary();
                // 1 slot for the left side, plus the right side's requirement
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            } else if (isImplicitMultiplicationTrigger(c)) {
                // No operator token to consume here -- e.g. "3t", "5sin(4)",
                // or "3'x'" -- treated exactly like an explicit '*' at this
                // same precedence tier. See isImplicitMultiplicationTrigger's
                // javadoc for what is (and deliberately is not) recognized.
                binaryOpCount++;
                int rightDepth = parseUnary();
                maxDepth = Math.max(maxDepth, 1 + rightDepth);
            } else {
                break;
            }
        }
        return maxDepth;
    }

    /**
     * Unary +, -, ! bind LESS tightly than '^' (matching standard math
     * convention: {@code -2^3 == -(2^3)}, not {@code (-2)^3}). This method
     * is therefore the entry point that decides whether to consume a sign
     * (recursing into itself to allow chains like "--3" or "-!x", bounded by
     * the nesting guard) or fall through to {@link #parsePower()}.
     *
     * <p>The Unicode root symbol {@code √} is treated as a fourth prefix
     * unary operator alongside {@code + - !}: {@code √9}, {@code -√9}, and
     * {@code √√9} are all handled by the same recursive chain.</p>
     *
     * <p>A run of one or more Unicode superscript digits immediately
     * preceding {@code √} (no whitespace in between) is recognized as an
     * "nth root" degree prefix, e.g. {@code ³√9} (cube root of 9) or
     * {@code ²√27} (square root of 27). This is distinct from the postfix
     * superscript exponent handled in {@link #parsePower()} — that one
     * follows a primary/power base ({@code x²}); this one precedes the root
     * symbol itself and is consumed as part of this method's prefix chain,
     * contributing depth exactly like the bare {@code √} case.</p>
     */
    private int parseUnary() {
        skipWhitespace();
        char c = peek();

        if (isSuperscriptDigit(c)) {
            int lookahead = pos;
            while (lookahead < expr.length() && isSuperscriptDigit(expr.charAt(lookahead))) {
                lookahead++;
            }
            if (lookahead < expr.length() && expr.charAt(lookahead) == ROOT_CHAR) {
                pos = lookahead; // consume the degree digit run
                nextChar();      // consume '√'
                unaryOpCount++;
                enterNesting();
                try {
                    int operandHeight = parseUnary();
                    return 1 + operandHeight;
                } finally {
                    exitNesting();
                }
            }
            // A superscript-digit run not immediately followed by '√' is not
            // valid as a prefix here (postfix superscripts are only ever
            // recognized after a primary/power base, in parsePower()); fall
            // through so parsePrimary() reports a precise "unexpected
            // character" error rather than silently doing nothing with it.
        }

        if (c == '+' || c == '-' || c == '!' || c == ROOT_CHAR) {
            nextChar();
            unaryOpCount++;
            enterNesting();
            try {
                int operandHeight = parseUnary();
                return 1 + operandHeight;
            } finally {
                exitNesting();
            }
        }
        return parsePower();
    }

    /**
     * Power's base is a plain primary (NOT unary) so that a leading sign at
     * this level is left for the caller ({@link #parseUnary()}) to wrap
     * around the entire power expression, e.g. "-2^3" -> -(2^3). The
     * exponent, however, is parsed via {@link #parseUnary()} so that
     * "2^-3" and right-associative chains like "2^3^2" both work correctly.
     *
     * <p>After the base is parsed, a run of one or more Unicode superscript
     * digits (⁰¹²³⁴⁵⁶⁷⁸⁹) is also recognized here as a postfix "raised to
     * the power of &lt;these digits&gt;" operator, e.g. {@code "x²"} ==
     * {@code "x^2"} and {@code "y²³"} == {@code "y^23"} (the whole run of
     * superscript digits forms a single literal exponent, contributing one
     * unary-style depth increment — there is no separately parsed exponent
     * subexpression, so it is counted like the other unary operators
     * rather than as a binary '^').</p>
     */
    private int parsePower() {
        int leftDepth = parsePrimary();
        skipWhitespace();

        if (isSuperscriptDigit(peek())) {
            while (isSuperscriptDigit(peek())) {
                nextChar();
            }
            unaryOpCount++;
            leftDepth = 1 + leftDepth;
            skipWhitespace();
        }

        if (peek() == '^') {
            nextChar();
            binaryOpCount++;
            int rightDepth = parseUnary(); // right-associative, allows a sign on the exponent
            // 1 slot for the left side, plus the right side's requirement
            return Math.max(leftDepth, 1 + rightDepth);
        }
        return leftDepth;
    }

    private int parsePrimary() {
        skipWhitespace();
        char c = peek();

        if (c == '\0') {
            throw new ParseException("Unexpected end of expression", pos);
        }

        // Number
        if (Character.isDigit(c) || c == '.') {
            int start = pos;
            consumeNumber();
            validateNumberToken(start, pos);
            return 1;
        }

        // String literal (single- or double-quoted), e.g. 'hello' or
        // "hello". A leaf exactly like a number or variable -- contributes
        // depth 1. See consumeStringLiteral() for escaping/termination rules.
        if (c == SINGLE_QUOTE || c == DOUBLE_QUOTE) {
            consumeStringLiteral(c);
            return 1;
        }

        // Parentheses: either plain grouping around a single expression, e.g.
        // "(1 + 2)", or a comma-separated literal list (matrix/vector data),
        // e.g. "(3,1,4,7)". Both are handled uniformly by
        // parseCommaListAndClose(): a single item behaves exactly like the
        // original "parens add no depth" grouping, while multiple items fall
        // back to the same stack-accumulation formula used for function args.
        if (c == '(') {
            nextChar();
            enterNesting();
            try {
                return parseCommaListAndClose("Missing closing ')'");
            } finally {
                exitNesting();
            }
        }

        // Anonymous function / matrix definition, e.g.:
        //   @(x)sin(x)                  -> anonymous function, no '=' before body
        //   @(x,y,z)=2*x+3*y+4*z        -> anonymous function, '=' before body
        //   @(2,2)(3,1,4,7)             -> anonymous matrix, dims then data
        //   @(2,2)=(3,1,4,7)            -> anonymous matrix, '=' before data
        if (c == '@') {
            return parseAnonymousDefinition();
        }

        // Variable or function
        if (Character.isLetter(c) && !isReservedOperatorChar(c)) {
            int nameStart = pos;
            String name = consumeIdentifier();
            skipWhitespace();
            if (peek() == '(') {
                nextChar(); // (
                functionCount++;
                boolean isDiffEq = DIFF_EQ_FUNCTIONS.contains(name);

                // ROOT-ONLY RULE for diffeqn / diffeqnPath / diffeqnHO /
                // diffeqnPathHO: a call to one of these functions may ONLY
                // appear as the entire (trimmed) input expression, optionally
                // wrapped in any number of pure grouping parentheses (the RPN
                // tokenizer downstream discards those anyway). Checked
                // structurally, right here, rather than as a post-hoc string
                // scan, so it composes correctly with every other feature.
                //
                // Legal:   diffeqn(...)         (((diffeqn(...))))
                // Illegal: 2+diffeqn(...), diffeqn(...)-2, sin(diffeqn(...)),
                //          diffeqn(...)*diffeqn(...), (diffeqn(...), 2)
                //
                // Any assignment prefix (e.g. "A = diffeqn(...)") is assumed
                // to have already been stripped by a preprocessor before this
                // string reaches this class.
                if (isDiffEq && !isOnlyWhitespaceAndChar(0, nameStart, '(')) {
                    throw new ParseException(
                            "A '" + name + "(...)' call must be the root expression "
                            + "(optionally wrapped in grouping parentheses); "
                            + "it cannot appear nested inside another expression.",
                            nameStart);
                }

                if (isDiffEq) {
                    diffEqDepth++;
                }
                enterNesting();
                try {
                    int result = parseCommaListAndClose(
                            "Missing closing ')' for function '" + name + "'");

                    if (isDiffEq) {
                        // Only whitespace and closing grouping parens may
                        // follow the call's own closing ')'.
                        if (!isOnlyWhitespaceAndChar(pos, expr.length(), ')')) {
                            throw new ParseException(
                                    "A '" + name + "(...)' call must be the root expression "
                                    + "(optionally wrapped in grouping parentheses); "
                                    + "nothing but closing parentheses may follow the closing ')'.",
                                    pos);
                        }
                    }

                    return result;
                } finally {
                    exitNesting();
                    if (isDiffEq) {
                        diffEqDepth--;
                    }
                }
            }
            // "name[n]" derivative-index notation, e.g. "y[3]" for y'''(x),
            // is recognized ONLY directly inside a call to one of the four
            // diffeqn-family functions (diffeqn/diffeqnPath/diffeqnHO/
            // diffeqnPathHO). Outside that context '[' is simply not
            // consumed here, so it falls through as an unrecognized
            // character (either "trailing content" or a later
            // ParseException from a nested parsePrimary()), exactly as
            // before this feature existed.
            if (diffEqDepth > 0 && peek() == '[') {
                nextChar(); // consume '['
                int digitsStart = pos;
                while (Character.isDigit(peek())) {
                    nextChar();
                }
                if (pos == digitsStart) {
                    throw new ParseException(
                            "Expected a non-negative integer derivative index inside '"
                            + name + "[...]'", pos);
                }
                skipWhitespace();
                if (peek() != ']') {
                    throw new ParseException(
                            "Missing closing ']' for derivative index on '" + name + "'", pos);
                }
                nextChar(); // consume ']'
                // "name[n]" denotes a single distinguished operand (the n-th
                // derivative of name), not an operation applied to name — it
                // pushes exactly one stack slot, same as a plain variable.
                return 1;
            }
            // plain variable pushes 1 item to stack
            return 1;
        }

        throw new ParseException("Unexpected character '" + c + "'", pos);
    }

    /**
     * Parses {@code '@' '(' header ')' [ '=' ] ( '(' data ')' | body )} —
     * i.e. everything after (and including) an already-peeked '@'.
     *
     * <p>The header (parameter names for a function, or dimensions for a
     * matrix) and, when present, the data list are both parsed with
     * {@link #parseCommaListAndClose(String)}, exactly like a function's
     * argument list. What follows the header determines the shape:</p>
     * <ul>
     *   <li>Another {@code '('} means an anonymous matrix literal: the
     *       depth is {@code max(headerDepth, dataDepth)}.</li>
     *   <li>Anything else is parsed as a single body expression (an
     *       anonymous function): the depth is
     *       {@code max(headerDepth, bodyDepth)}, and the call is counted
     *       toward {@link #functionCount} like a named function definition.</li>
     * </ul>
     * An optional single {@code '='} between the header and what follows is
     * tolerated either way (covers both {@code @(x)sin(x)} and
     * {@code @(x,y,z)=2*x+3*y+4*z} style bodies) — but a genuine {@code '=='}
     * is deliberately left untouched for the caller to interpret as equality.
     */
    private int parseAnonymousDefinition() {
        nextChar(); // consume '@'
        enterNesting();
        try {
            skipWhitespace();
            if (peek() != '(') {
                throw new ParseException(
                        "Expected '(' after '@' to start an anonymous function/matrix definition", pos);
            }
            nextChar();
            int headerDepth = parseCommaListAndClose(
                    "Missing closing ')' for anonymous function/matrix header");

            skipWhitespace();
            // Optional single '=' between the header and the body/data. Do NOT
            // consume it if it's actually the start of '==' (equality) —
            // extremely unlikely grammatically at this position, but cheap to
            // guard against and keeps this symmetric with the '=' handling in
            // parseRelationalLogical().
            if (peek() == '=' && !(pos + 1 < expr.length() && expr.charAt(pos + 1) == '=')) {
                nextChar();
                skipWhitespace();
            }

            if (peek() == '(') {
                // Anonymous matrix literal: header = dimensions, this = data.
                nextChar();
                int dataDepth = parseCommaListAndClose(
                        "Missing closing ')' for anonymous matrix data list");
                return Math.max(headerDepth, dataDepth);
            }

            // Anonymous function: header = parameter names, this = body.
            functionCount++;
            int bodyDepth = parseExpression();
            return Math.max(headerDepth, bodyDepth);
        } finally {
            exitNesting();
        }
    }

    /**
     * Parses a comma-separated list of expressions up to (and consuming) the
     * closing {@code ')'} — the current position must be immediately after
     * the already-consumed opening {@code '('}. Used for function-call
     * argument lists, plain grouping parens (a "list" of exactly one item),
     * parenthesized literal data (matrix/vector), and anonymous '@'
     * function/matrix headers and data lists — one implementation shared
     * across all of them so their depth accounting can never drift apart.
     *
     * <p>Depth accounting mirrors a stack machine building up the items
     * left-to-right: the {@code i}-th item (0-indexed) contributes
     * {@code i + depth(item_i)}, since the {@code i} previously-evaluated
     * items are still occupying stack slots while it is evaluated. The
     * overall result is the max of that across all items, or {@code 1} if
     * the list is empty (an empty {@code ()} still takes one slot to push
     * *something*, e.g. an empty-arg function call result).</p>
     *
     * @param missingCloseMessage exact message to use if the closing ')' is
     *                             missing, so callers can give context-specific
     *                             wording (e.g. naming the function).
     */
    private int parseCommaListAndClose(String missingCloseMessage) {
        int maxStackDepth = 0;
        int itemsCurrentlyOnStack = 0;

        skipWhitespace();
        boolean hasItems = peek() != ')';

        if (hasItems) {
            while (true) {
                int itemDepth = parseExpression();
                // The peak depth is the depth required to evaluate THIS item,
                // PLUS the items we've already evaluated and left on the stack.
                maxStackDepth = Math.max(maxStackDepth, itemsCurrentlyOnStack + itemDepth);
                itemsCurrentlyOnStack++;

                skipWhitespace();
                if (peek() == ',') {
                    nextChar();
                    skipWhitespace();
                } else {
                    break;
                }
            }
        }

        skipWhitespace();
        if (peek() != ')') {
            throw new ParseException(missingCloseMessage, pos);
        }
        nextChar();

        // An empty list, or a single item, still takes at least 1 slot.
        return maxStackDepth > 0 ? maxStackDepth : 1;
    }

    // ──────────────────────────────────────────────
    //  Token helpers
    // ──────────────────────────────────────────────
    private void consumeNumber() {
        while (Character.isDigit(peek())) {
            nextChar();
        }
        if (peek() == '.') {
            nextChar();
            while (Character.isDigit(peek())) {
                nextChar();
            }
        }
        char e = peek();
        if (e == 'e' || e == 'E') {
            nextChar();
            char sign = peek();
            if (sign == '+' || sign == '-') {
                nextChar();
            }
            while (Character.isDigit(peek())) {
                nextChar();
            }
        }
    }

    private void validateNumberToken(int start, int end) {
        String token = expr.substring(start, end);
        if (!NUMBER_PATTERN.matcher(token).matches()) {
            throw new ParseException("Invalid numeric literal '" + token + "'", start);
        }
    }

    /**
     * Consumes a string literal starting at the current position, which
     * must be positioned exactly on the opening quote character
     * {@code quoteChar} (either {@link #SINGLE_QUOTE} or
     * {@link #DOUBLE_QUOTE}). Advances {@link #pos} to just past the
     * matching, unescaped closing quote of the SAME kind — a single-quoted
     * string may freely contain unescaped double quotes and vice versa,
     * e.g. {@code "it's"} or {@code 'she said "hi"'}.
     *
     * <p>Backslash escaping is supported inside the literal: {@code \}
     * followed by any character is consumed as a single two-character unit
     * (so {@code \'}, {@code \"}, and {@code \\} all do what you'd expect,
     * and an escape of any other character is simply taken literally —
     * the backslash-escape pair is skipped over rather than rejected, the
     * same "don't guess, but don't gratuitously reject" balance the rest
     * of this grammar uses for run-together text). In particular this
     * guarantees a {@code \} immediately before the closing quote (e.g.
     * {@code "it\\"}, a literal trailing backslash) is handled correctly:
     * that backslash escapes itself, not the quote that follows it.</p>
     *
     * <p>An unescaped newline, or reaching end-of-input, before the
     * matching closing quote is found throws a {@link ParseException}
     * pointing at the opening quote — a string literal must be complete
     * and confined to a single line, exactly like every other token this
     * class recognizes (there is no multi-line/triple-quoted form).</p>
     *
     * @param quoteChar {@link #SINGLE_QUOTE} or {@link #DOUBLE_QUOTE};
     *                  whichever one opened this literal
     */
    private void consumeStringLiteral(char quoteChar) {
        int start = pos;
        nextChar(); // consume opening quote
        while (true) {
            char c = peek();
            if (c == '\0') {
                throw new ParseException(
                        "Unterminated string literal: missing closing " + quoteChar, start);
            }
            if (c == '\n' || c == '\r') {
                throw new ParseException(
                        "Unterminated string literal: unescaped line break before closing "
                        + quoteChar, start);
            }
            if (c == '\\') {
                nextChar(); // consume the backslash
                if (peek() == '\0') {
                    throw new ParseException(
                            "Unterminated string literal: dangling '\\' before end of input",
                            start);
                }
                nextChar(); // consume the escaped character, whatever it is
                continue;
            }
            if (c == quoteChar) {
                nextChar(); // consume closing quote
                return;
            }
            nextChar(); // ordinary content character, including the OTHER quote kind
        }
    }

    private String consumeIdentifier() {
        int start = pos;
        while ((Character.isLetterOrDigit(peek()) || peek() == '_') && !isReservedOperatorChar(peek())) {
            nextChar();
        }
        return expr.substring(start, pos);
    }

    private char peek() {
        return pos < expr.length() ? expr.charAt(pos) : '\0';
    }

    private char nextChar() {
        return pos < expr.length() ? expr.charAt(pos++) : '\0';
    }




    // ══════════════════════════════════════════════════════════════════════
    //  TOKEN-ARRAY BASED DEPTH CALCULATION
    // ══════════════════════════════════════════════════════════════════════
    /**
     * Computes the AST depth (height) directly from an already-compiled
     * token stream, as produced by {@code MathExpression}'s tokenizer/compiler
     * for the bulk (SIMD / GPU) evaluator, instead of re-lexing a raw string.
     *
     * <p><b>Token order: this is POSTFIX (Reverse Polish), not infix.</b>
     * The bulk evaluator pipeline hands this method the same linear,
     * already-shunting-yarded stream it will itself walk with a stack
     * machine at evaluation time — e.g. {@code 3 + 2*x} arrives as
     * {@code [3, 2, x, *, +]}, not {@code [3, +, 2, *, x]}. This method
     * performs the correct thing for postfix input: a linear left-to-right
     * <b>stack simulation</b>, mirroring exactly how the real evaluator will
     * execute the stream.</p>
     *
     * <p>Simulation rules (kept in lockstep with the char-based
     * {@link #calculate()} above):
     * <ul>
     *   <li>NUMBER / VARIABLE / MATRIX / STRING / FUNCTION_HANDLE(_UNDEFINED) — push
     *       depth 1. Anonymous '@' function/matrix definitions are expected to already
     *       be compiled down to one of these leaf kinds by the tokenizer by
     *       the time they reach the bulk evaluator, the same way METHOD calls
     *       arrive with their {@code rawArgs} already resolved.
     *       <b>NOTE:</b> this assumes {@code MathExpression.Token} exposes a
     *       {@code STRING} kind constant, parallel to {@code NUMBER}/
     *       {@code VARIABLE}/{@code MATRIX}, that the compiler emits for a
     *       single- or double-quoted string literal. Add that constant to
     *       {@code MathExpression.Token} alongside this change if it does not
     *       already exist — a string leaf otherwise has no way to be
     *       represented in the compiled postfix stream.</li>
     *   <li>OPERATOR with arity 1 (prefix or postfix unary, e.g. {@code -x}, {@code x!},
     *       {@code √x}, or a compiled Unicode-superscript power) —
     *       pop 1 depth {@code d}, push {@code 1 + d}.</li>
     *   <li>OPERATOR with arity 2 (binary infix, e.g. {@code +  *  ^  <  &&}) —
     *       pop 2 depths {@code d1, d2} (evaluation order irrelevant here since
     *       the combination is symmetric), push {@code 1 + max(d1, d2)}.</li>
     *   <li>FUNCTION / METHOD with arity {@code k} — pop {@code k} depths in
     *       their original left-to-right argument order and push
     *       {@code max(i + argDepth[i])} for {@code i} in {@code [0, k)}
     *       (never less than 1). This is the same "arguments already
     *       evaluated and left on the stack" accounting the char-based
     *       function-call handling above uses.</li>
     *   <li>LPAREN / RPAREN / COMMA — not expected to appear in a compiled
     *       postfix stream (those are purely infix-source punctuation that
     *       shunting-yard consumes); encountering one is treated as a
     *       malformed stream.</li>
     * </ul>
     *
     * <p>This method is strict: an empty stack when an operator/function
     * needs to pop operands, more than one value left on the stack at the
     * end, an unexpected token kind, or a {@code null} element all throw
     * {@link IllegalArgumentException} carrying the offending token index —
     * never a best-effort guess. A silently-wrong depth is far more
     * dangerous to a SIMD/GPU evaluator — which typically pre-sizes fixed
     * register/stack buffers off this number — than a loud, early failure.</p>
     *
     * @param tokens the compiled token stream in postfix (RPN) order, as
     *               handed to the bulk evaluator. May be {@code null} or
     *               empty, in which case a zero-depth {@link Result} is returned.
     * @return a {@link Result} describing the computed depth plus the same
     *         operator/function counters produced by {@link #calculate()}.
     * @throws IllegalArgumentException if the token stream is structurally
     *         invalid (stack underflow, leftover values, unexpected token
     *         kind, {@code null} token, etc.)
     */
    public static Result calculate(MathExpression.Token[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return new Result(0, 0, 0, 0, 0);
        }

        int[] stack = new int[Math.max(8, tokens.length)];
        int sp = 0; // stack pointer: number of items currently on the stack

        int binaryOpCount = 0;
        int divOpCount = 0;
        int unaryOpCount = 0;
        int functionCount = 0;

        for (int i = 0; i < tokens.length; i++) {
            MathExpression.Token t = tokens[i];
            if (t == null) {
                throw new IllegalArgumentException("Null token element found at index " + i);
            }

            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                case MathExpression.Token.VARIABLE:
                case MathExpression.Token.MATRIX:
                case MathExpression.Token.STRING:
                case MathExpression.Token.FUNCTION_HANDLE:
                case MathExpression.Token.FUNCTION_HANDLE_UNDEFINED:
                    stack[sp++] = 1;
                    break;

                case MathExpression.Token.OPERATOR: {
                    if (t.arity == 1) {
                        if (sp < 1) {
                            throw new IllegalArgumentException(
                                    "Stack underflow at index " + i + ": unary OPERATOR '" + t.opChar
                                    + "' needs 1 operand but the stack is empty "
                                    + "(malformed postfix stream)");
                        }
                        int d = stack[--sp];
                        unaryOpCount++;
                        stack[sp++] = 1 + d;
                    } else if (t.arity == 2) {
                        if (sp < 2) {
                            throw new IllegalArgumentException(
                                    "Stack underflow at index " + i + ": binary OPERATOR '" + t.opChar
                                    + "' needs 2 operands but only " + sp + " value(s) are on the stack "
                                    + "(malformed postfix stream)");
                        }
                        int d2 = stack[--sp];
                        int d1 = stack[--sp];
                        binaryOpCount++;
                        if (t.opChar == '/') {
                            divOpCount++;
                        }
                        stack[sp++] = 1 + Math.max(d1, d2);
                    } else {
                        throw new IllegalArgumentException(
                                "Unsupported OPERATOR arity " + t.arity + " for '" + t.opChar
                                + "' at index " + i + " (expected 1 or 2)");
                    }
                    break;
                }

                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD: {
                    int arity = t.arity;
                    if (sp < arity) {
                        throw new IllegalArgumentException(
                                "Stack underflow at index " + i + ": call to '" + t.name + "' needs "
                                + arity + " argument(s) but only " + sp + " value(s) are on the stack "
                                + "(malformed postfix stream)");
                    }
                    // Pop the arity depths off; they come off in reverse (last
                    // argument first), so write them back into left-to-right
                    // argument order before applying the stack-accumulation formula.
                    int maxStackDepth = 0;
                    for (int argIndex = arity - 1; argIndex >= 0; argIndex--) {
                        int argDepth = stack[--sp];
                        maxStackDepth = Math.max(maxStackDepth, argIndex + argDepth);
                    }
                    functionCount++;
                    // Empty call e.g. rand() still takes 1 slot to push the result
                    stack[sp++] = maxStackDepth > 0 ? maxStackDepth : 1;
                    break;
                }

                case MathExpression.Token.LPAREN:
                case MathExpression.Token.RPAREN:
                case MathExpression.Token.COMMA:
                    throw new IllegalArgumentException(
                            "Unexpected " + MathExpression.Token.getKind(t.kind) + " token at index " + i
                            + " in a postfix token stream (parentheses/commas are infix-source "
                            + "punctuation and should not appear in compiled RPN output)");

                default:
                    throw new IllegalArgumentException(
                            "Unexpected token kind " + MathExpression.Token.getKind(t.kind)
                            + " at index " + i);
            }

            if (sp >= stack.length) {
                // Should be unreachable for a well-formed stream (sp can never
                // exceed tokens.length), but guard against a hostile/corrupt
                // arity value driving unbounded growth instead of throwing
                // an opaque ArrayIndexOutOfBoundsException deep in the loop.
                int[] grown = new int[stack.length * 2];
                System.arraycopy(stack, 0, grown, 0, stack.length);
                stack = grown;
            }
        }

        if (sp != 1) {
            throw new IllegalArgumentException(
                    "Malformed postfix token stream: expected exactly 1 value left on the stack "
                    + "after processing all " + tokens.length + " token(s), found " + sp
                    + " (missing operator(s) between operands, or a truncated stream)");
        }

        return new Result(stack[0], binaryOpCount, divOpCount, unaryOpCount, functionCount);
    }

    // ──────────────────────────────────────────────
    //  Demo
    // ──────────────────────────────────────────────
    public static void main(String[] args) {
        String[] tests = {
            "x",
            "2 + 3 * -4 ^ 2",
            "-2 + --3",
            "sin(2 + cos(x)) + max(a, b, 3)",
            "2^-3 + log10(1e-4 * y)",
            "(1 + (2 + (3 + 4)))",
            "-2^3",
            "if(x-y>7 && 3*x-5<=8, x*(3-y), 12)",
            "orange + 3",             // must NOT be misparsed as "OR ange"
            "a AND b OR c",           // textual keyword operators, space-delimited
            "!isValid && x > 2",      // unary logical NOT
            "sin (x)",                // whitespace before a function call's '(' must still work
            "aANDbORc",               // no spaces at all -> a single valid identifier, NOT keywords
            "f=@(x)sin(x)",           // anonymous function, no '=' before body
            "g=@(x,y,z)=2*x+3*y+4*z", // anonymous function, '=' before body
            "f(x)=sin(x)",            // named function definition shorthand
            "g(x,y,z)=2*x+3*y+4*z",   // named function definition shorthand, multi-arg
            "A(2,2)=(3,1,4,7)",       // named matrix definition, parenthesized data list
            "A=@(2,2)(3,1,4,7)",      // anonymous matrix, dims then data
            "A=@(2,2)=(3,1,4,7)",     // anonymous matrix, '=' before data
            "2\u00B2+3\u00B3+\u221A9",         // "2²+3³+√9" -- Unicode superscripts + root
            "-\u221A9",                        // "-√9" -- signed root
            "\u221A\u221A9",                   // "√√9" -- nested root
            "x\u00B2\u00B3",                   // "x²³" -- multi-digit superscript exponent (== x^23)
            "\u00B3\u221A9",                   // "³√9" -- prefix cube root
            "-\u00B3\u221A27 + \u00B2\u221A4",  // "-³√27 + ²√4" -- signed prefix root, mixed with square root
            "diffeqn((3*x^2)*y[4]+(5*sin(x))*y[3]+(5/x)*y[2]-3*y[1]+3*x*y[0], other_args)",
            "diffeqnHO(y[2]+y[0], a, b)",       // y[n] also recognized in the other diffeqn-family calls
            "diffeqn(x[0] + sin(y[3]), 1)",     // y[n] still recognized nested inside sin(...) within the call
            "9\u04203",                          // "9Р3" -- nPr permutation
            "6\u010C5",                          // "6Č5" -- nCr combination
            "5! + 9\u04203 + 6\u010C5",           // the reported crashing expression
            "(a+b)\u0420(c-d) * 2\u010C1",       // combinatoric operators combined with grouping/multiplication
            "3t",                                 // implicit multiplication: "3t" == "3*t"
            "5sin(4)",                             // implicit multiplication: "5sin(4)" == "5*sin(4)"
            "3(x+1)",                              // implicit multiplication: number juxtaposed with a group
            "(x+1)(x-1)",                          // implicit multiplication between two parenthesized groups
            "2\u221A9",                            // implicit multiplication: number juxtaposed with '√'
            "diffeqnHO((3t^2)*y[4]+(5*sin(t))*y[3]+(5/t)*y[2]-3*y[1]+3*t*y[0], 0, y0, 20, 0.01, rk4)",
            "diffeqnHO((3t^2)*y[4]+(5*sin(t))*y[3]+(5/t)*y[2]-3*y[1]+3*t*y[0], 0, @(1,5)(1, 0, 0, 0, 0), 20, 0.01, rk4)",
            // the equation from the reported crash, now accepted via implicit multiplication
            "  diffeqn(y[1]+y[0], a, b)  ",     // legal root call: surrounding whitespace only
            "(((diffeqn(y[1]+y[0], a, b))))",   // legal: any number of pure grouping parens around the root call
            "( diffeqn(y[0], a) )",             // legal: single grouping wrap, with inner/outer whitespace
            "(diffeqn(y[0], a))",               // legal: single grouping wrap, no extra whitespace
            "diffeqnPathHO(y[2]+y[0], a, b, c)", // the fourth diffeqn-family name, as a bare root call
            "((diffeqnPathHO(y[2]+y[0], a, b, c)))", // same, wrapped in grouping parens
            "'hello'",                          // bare single-quoted string literal
            "\"hello\"",                        // bare double-quoted string literal
            "concat('a', \"b\")",               // string literals as function arguments, mixed quote styles
            "'it''s complicated'",              // a double single-quote is just two adjacent characters, not an escape
            "'it\\'s escaped'",                 // backslash-escaped single quote inside a single-quoted string
            "\"she said \\\"hi\\\"\"",           // backslash-escaped double quote inside a double-quoted string
            "'she said \"hi\" inline'",         // unescaped double quotes freely nested inside a single-quoted string
            "\"trailing backslash: \\\\\"",      // literal trailing backslash, correctly not escaping the closing quote
            "'' + \"\"",                         // two empty string literals, added together
            "3'x'",                              // implicit multiplication: number juxtaposed with a string
            "'a'x",                              // implicit multiplication: string juxtaposed with an identifier
            "(x+1)'y'",                          // implicit multiplication: group juxtaposed with a string
            "len('hello') + len(\"world\")",     // strings as args to two different function calls, then summed
        };

        for (String s : tests) {
            try {
                Result r = new MathExpressionTreeDepth(s).calculate();
                System.out.printf("%-45s -> %s%n", s, r);
            } catch (ParseException pe) {
                System.out.printf("%-45s -> PARSE ERROR: %s%n", s, pe.getMessage());
            }
        }

        // Deliberately malformed input, to demonstrate graceful failure
        // instead of a silently wrong/incomplete depth.
        String[] malformed = {
            "2 + 3)",        // unbalanced parenthesis / trailing content
            "(1 + 2",        // missing closing parenthesis
            "sin(x",         // missing closing parenthesis on a function
            "5.",            // incomplete numeric literal
            "2 & $ 3",       // unrecognized character
            "1 2",           // two operands with no operator between them
            "3ORx",          // ambiguous run-together text -> rejected rather than guessed at
            "@x",            // '@' not followed by '('
            "A=@(2,2)(3,1,4", // missing closing ')' on the data list
            "sin[x]",        // '[' is not a general grouping/function delimiter outside diffeqn-family calls
            "y[3] + 1",      // "name[n]" outside any diffeqn-family call is rejected
            "diffeqn(y[-1], a)", // negative index is not a non-negative integer
            "diffeqn(y[3, a)",   // missing closing ']' on the derivative index
            "\u04203 + 1",        // permutation operator with no left operand
            "2+diffeqn(y[0]+y[1], a)",          // diffeqn NOT the root: something precedes it
            "diffeqn(y[0]+y[1], a)-2",          // diffeqn NOT the root: something follows it
            "sin(diffeqn(y[0], a))",            // diffeqn nested inside another function call
            "diffeqn(y[0],a)*diffeqnHO(y[1],b)",// two diffeqn-family calls: neither is a lone root
            "-diffeqn(y[0], a)",                // diffeqn preceded by a unary sign
            "diffeqnPath(diffeqn(y[0], a), b)", // one diffeqn-family call nested inside another
            "(diffeqn(y[0], a), 2)",             // a two-item list, NOT a pure grouping wrap
            "(diffeqn(y[0], a))+1",              // pure-wrapped call still followed by an operator
            "((diffeqn(y[0], a))",               // unbalanced: extra unmatched leading '('
            "1+diffeqnPathHO(y[2]+y[0], a, b, c)",       // fourth name, nested (something precedes it)
            "diffeqnPathHO(y[2]+y[0], a, b, c)-1",       // fourth name, followed by an operator
            "cos(diffeqnPathHO(y[2]+y[0], a, b, c))",    // fourth name, nested inside another function
            "'unterminated",                     // missing closing single quote
            "\"unterminated",                    // missing closing double quote
            "'bad escape at end\\",              // dangling backslash right before end of input
            "'line\nbreak'",                     // unescaped newline inside a string literal
        };

        System.out.println("\n-- malformed input (expected to fail cleanly) --");
        for (String s : malformed) {
            try {
                Result r = new MathExpressionTreeDepth(s).calculate();
                System.out.printf("%-45s -> %s (unexpectedly succeeded!)%n", s, r);
            } catch (ParseException pe) {
                System.out.printf("%-45s -> PARSE ERROR: %s%n", s, pe.getMessage());
            }
        }
    }
}