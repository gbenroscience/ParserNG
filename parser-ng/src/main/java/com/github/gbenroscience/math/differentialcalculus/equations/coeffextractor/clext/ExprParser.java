package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

import java.util.ArrayList;
import java.util.List;

/**
 * Recursive-descent parser over the flat token vocabulary produced by
 * {@link Lexer} (or supplied directly as an already-scanned token list),
 * producing an {@link ExprNode} tree.
 *
 * Grammar (standard precedence, lowest to highest):
 * <pre>
 *   expression := term (('+' | '-') term)*
 *   term       := unary (('*' | '/') unary)*
 *   unary      := '-' unary | power
 *   power      := primary ('^' unary)?              // right-associative
 *   primary    := NUMBER
 *               | IDENT '(' argList ')'              // function call, any arity
 *               | IDENT '[' expression ']'           // indexed access — see below
 *               | IDENT                              // bare variable
 *               | '(' expression ')'
 *   argList    := expression (',' expression)* | (empty)
 * </pre>
 *
 * <h2>Indexed access — any identifier, by default</h2>
 * This parser is not a general reimplementation of ParserNG's grammar — its
 * only reason to exist is feeding CoefficientExtractor. That said, the
 * dependent variable in a differential equation is not always called "y" —
 * so by default ({@link #parse(List)}), indexed access (IDENT '[' ... ']')
 * is accepted for <em>any</em> identifier, producing a state-variable leaf
 * via {@link ExprNode#stateVariable} named after whatever identifier was
 * actually used. Downstream, {@link CoefficientExtractor} is responsible for
 * checking that only one distinct name is actually used as a state variable
 * across the whole equation (see its javadoc) — this parser does not enforce
 * that itself, since a single expression tree in isolation has no way to
 * know whether a name mismatch is a real error or intentional.
 *
 * If the caller already knows which name to expect, {@link #parse(List, String)}
 * accepts a specific state-variable name and rejects indexed access on any
 * other identifier immediately, at the point it's encountered — earlier and
 * more specific than discovering a mismatch after full parsing.
 *
 * In both cases, the bracketed index expression must reduce to a
 * non-negative integer constant (a literal number, or a constant-folded
 * arithmetic expression of literals) — a symbolic or non-constant index
 * cannot be statically assigned to a fixed derivative-order slot, so it is
 * rejected with a clear message rather than silently mishandled.
 */
public final class ExprParser {

    private final List<String> tokens;
    /** null means "any identifier may be indexed" (auto-detect mode); non-null restricts to that one name. */
    private final String requiredStateVarNameOrNull;
    private int pos;

    private ExprParser(List<String> tokens, String requiredStateVarNameOrNull) {
        this.tokens = tokens;
        this.requiredStateVarNameOrNull = requiredStateVarNameOrNull;
        this.pos = 0;
    }

    /** Auto-detect mode: any identifier may be indexed; each becomes a state-variable leaf under its own name. */
    public static ExprNode parse(List<String> tokens) {
        return parseInternal(tokens, null);
    }

    /** Restrictive mode: only requiredStateVarName may be indexed; any other indexed identifier is rejected immediately. */
    public static ExprNode parse(List<String> tokens, String requiredStateVarName) {
        if (requiredStateVarName == null) {
            throw new IllegalArgumentException(
                    "requiredStateVarName must not be null — use parse(tokens) for auto-detect mode instead.");
        }
        return parseInternal(tokens, requiredStateVarName);
    }

    private static ExprNode parseInternal(List<String> tokens, String requiredStateVarNameOrNull) {
        if (tokens == null || tokens.isEmpty()) {
            throw new IllegalArgumentException("tokens must not be null or empty");
        }
        ExprParser parser = new ExprParser(tokens, requiredStateVarNameOrNull);
        ExprNode result = parser.parseExpression();
        if (parser.pos != parser.tokens.size()) {
            throw new IllegalArgumentException(
                    "Unexpected trailing token '" + parser.peek() + "' at position " + parser.pos
                    + " — expression ended but tokens remain.");
        }
        return result;
    }

    // ------------------------------------------------------------------

    private ExprNode parseExpression() {
        ExprNode left = parseTerm();
        while (check("+") || check("-")) {
            String op = advance();
            ExprNode right = parseTerm();
            left = ExprNode.op(op.charAt(0), List.of(left, right));
        }
        return left;
    }

    private ExprNode parseTerm() {
        ExprNode left = parseUnary();
        while (check("*") || check("/")) {
            String op = advance();
            ExprNode right = parseUnary();
            left = ExprNode.op(op.charAt(0), List.of(left, right));
        }
        return left;
    }

    private ExprNode parseUnary() {
        if (check("-")) {
            advance();
            ExprNode operand = parseUnary();
            return ExprNode.op('-', List.of(operand));
        }
        return parsePower();
    }

    private ExprNode parsePower() {
        ExprNode base = parsePrimary();
        if (check("^")) {
            advance();
            ExprNode exponent = parseUnary(); // right-associative: x^-y and x^y^z bind from the right
            return ExprNode.op('^', List.of(base, exponent));
        }
        return base;
    }

    private ExprNode parsePrimary() {
        if (check("(")) {
            advance();
            ExprNode inner = parseExpression();
            expect(")");
            return inner;
        }

        String tok = peek();
        if (tok == null) {
            throw new IllegalArgumentException("Unexpected end of input while parsing an expression.");
        }

        if (isNumber(tok)) {
            advance();
            return ExprNode.number(Double.parseDouble(tok));
        }

        if (isIdentifier(tok)) {
            advance();
            if (check("(")) {
                advance();
                List<ExprNode> args = parseArgList();
                expect(")");
                return ExprNode.func(tok, args);
            }
            if (check("[")) {
                if (requiredStateVarNameOrNull != null && !tok.equals(requiredStateVarNameOrNull)) {
                    throw new IllegalArgumentException(
                            "Indexed access '" + tok + "[...]' is not supported — only the state variable '"
                            + requiredStateVarNameOrNull + "' may be indexed here.");
                }
                advance();
                ExprNode indexExpr = parseExpression();
                expect("]");
                int index = requireConstantNonNegativeInteger(indexExpr, tok);
                return ExprNode.stateVariable(tok, index);
            }
            return ExprNode.variable(tok);
        }

        throw new IllegalArgumentException("Unexpected token '" + tok + "' at position " + pos + ".");
    }

    private List<ExprNode> parseArgList() {
        List<ExprNode> args = new ArrayList<>();
        if (check(")")) {
            return args; // empty argument list
        }
        args.add(parseExpression());
        while (check(",")) {
            advance();
            args.add(parseExpression());
        }
        return args;
    }

    /**
     * Reduces a bracket-index expression to a non-negative integer, or
     * throws. Only literal numbers and constant arithmetic over them are
     * accepted (e.g. "3", "1+1") — a symbolic index like y[k] where k is a
     * variable cannot be assigned to a fixed derivative-order slot.
     */
    private static int requireConstantNonNegativeInteger(ExprNode node, String indexedName) {
        double value = constantFold(node, indexedName);
        if (value < 0 || value != Math.floor(value) || Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException(
                    "Index inside " + indexedName + "[...] must be a non-negative integer constant, got " + value);
        }
        return (int) value;
    }

    private static double constantFold(ExprNode node, String indexedName) {
        switch (node.kind) {
            case NUMBER:
                return node.numberValue;
            case OP:
                if (node.funcName != null) {
                    break; // function calls are not constant-foldable here
                }
                if (node.children.size() == 1) {
                    return -constantFold(node.children.get(0), indexedName);
                }
                double a = constantFold(node.children.get(0), indexedName);
                double b = constantFold(node.children.get(1), indexedName);
                switch (node.opChar) {
                    case '+': return a + b;
                    case '-': return a - b;
                    case '*': return a * b;
                    case '/': return a / b;
                    case '^': return Math.pow(a, b);
                    default: break;
                }
                break;
            default:
                break;
        }
        throw new IllegalArgumentException(
                "Index inside " + indexedName + "[...] must be a constant (numbers and +,-,*,/,^ over "
                + "them only) — found a non-constant sub-expression.");
    }

    // ------------------------------------------------------------------

    private String peek() {
        return pos < tokens.size() ? tokens.get(pos) : null;
    }

    private boolean check(String expected) {
        String tok = peek();
        return tok != null && tok.equals(expected);
    }

    private String advance() {
        String tok = tokens.get(pos);
        pos++;
        return tok;
    }

    private void expect(String expected) {
        if (!check(expected)) {
            throw new IllegalArgumentException(
                    "Expected '" + expected + "' at position " + pos + " but found '" + peek() + "'.");
        }
        advance();
    }

    private static boolean isNumber(String tok) {
        return !tok.isEmpty() && (Character.isDigit(tok.charAt(0)));
    }

    private static boolean isIdentifier(String tok) {
        return !tok.isEmpty() && (Character.isLetter(tok.charAt(0)) || tok.charAt(0) == '_');
    }
}