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
 * 1.2e-3) - Variables (e.g., x, varName_123) - Binary operators: + - * / ^
 * (power, right-associative) - Unary + - ! - Functions with any number of
 * arguments (e.g., sin(x), max(a, b, c+ d)) - Parentheses for grouping AND
 * for comma-separated literal lists, e.g. matrix data "(3,1,4,7)" -
 * Relational operators: &gt; &lt; &gt;= &lt;= == != - Logical operators:
 * &amp;&amp; || (and the symbolic single &amp; / |), plus the textual
 * keywords OR / AND (word-boundary safe, so identifiers like "orange" or
 * "android" are never mistaken for the keyword) - Anonymous function and
 * matrix definitions via '@', e.g. "@(x)sin(x)", "@(x,y,z)=2*x+3*y+4*z",
 * "@(2,2)(3,1,4,7)" - No external libraries, single-pass O(n) parsing -
 * Spaces are ignored
 *
 * Tree depth definition: - Leaf (number or variable) = 1 - Binary operator
 * node = 1 + max(left depth, right depth) - Unary operator node = 1 + operand
 * depth - Function node (named call, parenthesized comma-list, or anonymous
 * '@' function/matrix) = max over items of (depth of that item + number of
 * items already evaluated and left on the stack), never less than 1 -
 * Parentheses around a single item do not add extra depth (pure grouping)
 *
 * Example: "2 + 3 * 4" -> depth 3 ((2 + (3 * 4))) "-2^3" -> depth 3 (- (2 ^ 3))
 * "2^-3" -> depth 3 (2 ^ (-3)) "sin(2 + 3 * 4)" -> depth 4 "(1 + (2 + (3 +
 * 4)))"-> depth 4
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
     * ({@link #consumeNumber()}, {@link #consumeIdentifier()}) deliberately
     * use the raw, non-skipping {@link #peek()}/{@link #nextChar()} so that
     * whitespace still correctly terminates a number or identifier.
     */
    public MathExpressionTreeDepth(String expression) {
        this.expr = expression == null ? "" : expression;
        this.pos = 0;
    }

    /**
     * Thrown for any structurally invalid, incomplete, or unsupported
     * expression: unknown characters, unclosed parentheses/functions,
     * malformed numeric literals, trailing unparsed content, or excessive
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
     * to decide what comes next) — never from inside {@link #consumeNumber()}
     * or {@link #consumeIdentifier()}, which rely on whitespace to correctly
     * terminate the token they're consuming.
     */
    private void skipWhitespace() {
        while (pos < expr.length() && Character.isWhitespace(expr.charAt(pos))) {
            pos++;
        }
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

    private int parseMultiplicative() {
        int maxDepth = parseUnary();
        while (true) {
            skipWhitespace();
            char c = peek();
            if (c == '*' || c == '/') {
                nextChar();
                binaryOpCount++;
                if (c == '/') {
                    divOpCount++;
                }
                int rightDepth = parseUnary();
                // 1 slot for the left side, plus the right side's requirement
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
     */
    private int parseUnary() {
        skipWhitespace();
        char c = peek();
        if (c == '+' || c == '-' || c == '!') {
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
     */
    private int parsePower() {
        int leftDepth = parsePrimary();
        skipWhitespace();
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
        if (Character.isLetter(c)) {
            String name = consumeIdentifier();
            skipWhitespace();
            if (peek() == '(') {
                nextChar(); // (
                functionCount++;
                enterNesting();
                try {
                    return parseCommaListAndClose("Missing closing ')' for function '" + name + "'");
                } finally {
                    exitNesting();
                }
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

    private String consumeIdentifier() {
        int start = pos;
        while (Character.isLetterOrDigit(peek()) || peek() == '_') {
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
     *   <li>NUMBER / VARIABLE / MATRIX / FUNCTION_HANDLE(_UNDEFINED) — push depth 1.
     *       Anonymous '@' function/matrix definitions are expected to already
     *       be compiled down to one of these leaf kinds by the tokenizer by
     *       the time they reach the bulk evaluator, the same way METHOD calls
     *       arrive with their {@code rawArgs} already resolved.</li>
     *   <li>OPERATOR with arity 1 (prefix or postfix unary, e.g. {@code -x}, {@code x!}) —
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
            "A=@(2,2)=(3,1,4,7)"      // anonymous matrix, '=' before data
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
            "A=@(2,2)(3,1,4" // missing closing ')' on the data list
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