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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

/**
 *
 * @author GBEMIRO
 */ 

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
 * <h2>Indexed access is intentionally narrow</h2>
 * This parser is not a general reimplementation of ParserNG's grammar — its
 * only reason to exist is feeding CoefficientExtractor. Indexed access
 * (IDENT '[' ... ']') is therefore only accepted when IDENT equals the
 * configured state-variable name (default "y"): that produces a state-
 * variable leaf via {@link ExprNode#stateVariable}, and the bracketed
 * expression must reduce to a non-negative integer constant (a literal
 * number, or a constant-folded arithmetic expression of literals) — a
 * symbolic or non-constant index cannot be statically assigned to a fixed
 * derivative-order slot, so it is rejected with a clear message rather than
 * silently mishandled. Indexed access on any other identifier is rejected as
 * out of scope for this parser.
 */
public final class ExprParser {

    private final List<String> tokens;
    private final String stateVarName;
    private int pos;

    private ExprParser(List<String> tokens, String stateVarName) {
        this.tokens = tokens;
        this.stateVarName = stateVarName;
        this.pos = 0;
    }

    /** Parses using "y" as the state-variable name. */
    public static ExprNode parse(List<String> tokens) {
        return parse(tokens, "y");
    }

    public static ExprNode parse(List<String> tokens, String stateVarName) {
        if (tokens == null || tokens.isEmpty()) {
            throw new IllegalArgumentException("tokens must not be null or empty");
        }
        ExprParser parser = new ExprParser(tokens, stateVarName);
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
                if (!tok.equals(stateVarName)) {
                    throw new IllegalArgumentException(
                            "Indexed access '" + tok + "[...]' is not supported — only the state variable '"
                            + stateVarName + "' may be indexed in this parser.");
                }
                advance();
                ExprNode indexExpr = parseExpression();
                expect("]");
                int index = requireConstantNonNegativeInteger(indexExpr, tok);
                return ExprNode.stateVariable(stateVarName, index);
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
    private static int requireConstantNonNegativeInteger(ExprNode node, String stateVarName) {
        double value = constantFold(node, stateVarName);
        if (value < 0 || value != Math.floor(value) || Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException(
                    "Index inside " + stateVarName + "[...] must be a non-negative integer constant, got " + value);
        }
        return (int) value;
    }

    private static double constantFold(ExprNode node, String stateVarName) {
        switch (node.kind) {
            case NUMBER:
                return node.numberValue;
            case OP:
                if (node.funcName != null) {
                    break; // function calls are not constant-foldable here
                }
                if (node.children.size() == 1) {
                    return -constantFold(node.children.get(0), stateVarName);
                }
                double a = constantFold(node.children.get(0), stateVarName);
                double b = constantFold(node.children.get(1), stateVarName);
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
                "Index inside " + stateVarName + "[...] must be a constant (numbers and +,-,*,/,^ over "
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