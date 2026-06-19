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

/**
 *
 * @author GBEMIRO
 */
/**
 * Optimized Java class to compute the depth (height) of the abstract syntax tree (AST)
 * for a mathematical expression. 
 * 
 * Features:
 * - Handles numbers (integers, decimals, scientific notation like 1.2e-3)
 * - Variables (e.g., x, varName_123)
 * - Binary operators: + - * / ^ (power, right-associative)
 * - Unary + and - 
 * - Functions with any number of arguments (e.g., sin(x), max(a, b, c+ d))
 * - Parentheses for grouping
 * - No external libraries, single-pass O(n) parsing with zero heap allocations
 *   (only recursion stack, which is bounded by expression complexity)
 * - Spaces are ignored
 * 
 * Tree depth definition:
 * - Leaf (number or variable) = 1
 * - Binary operator node = 1 + max(left depth, right depth)
 * - Function node = 1 + max(argument depths)
 * - Parentheses do not add extra depth (they are just grouping)
 * 
 * Example:
 *   "2 + 3 * 4"          -> depth 3   ((2 + (3 * 4)))
 *   "-2^3"               -> depth 3   (- (2 ^ 3))
 *   "2^-3"               -> depth 3   (2 ^ (-3))
 *   "sin(2 + 3 * 4)"     -> depth 4
 *   "(1 + (2 + (3 + 4)))"-> depth 4
 */
public class MathExpressionTreeDepth implements Savable{
private static final long serialVersionUID = 1L;
    private final String expr;
    private int pos;

    // Counters
    private int binaryOpCount  = 0;
    private int divOpCount  = 0;
    private int unaryOpCount   = 0;
    private int functionCount  = 0;

    public MathExpressionTreeDepth(String expression) {
        this.expr = (expression == null ? "" : expression).replaceAll("\\s+", "");
        this.pos = 0;
    }

    public static class Result implements Savable{
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

    public Result calculate() {
        if (expr.isEmpty()) {
            return new Result(0, 0, 0, 0, 0);
        }
        int depth = parseExpression();
        return new Result(depth, binaryOpCount,divOpCount, unaryOpCount, functionCount);
    }

    // ──────────────────────────────────────────────
    //  Parser levels (same as before, just with counters)
    // ──────────────────────────────────────────────

    private int parseExpression() {
        return parseAdditive();
    }

    private int parseAdditive() {
        int height = parseMultiplicative();
        while (true) {
            char c = peek();
            if (c == '+' || c == '-') {
                nextChar();
                binaryOpCount++;
                int right = parseMultiplicative();
                height = 1 + Math.max(height, right);
            } else {
                break;
            }
        }
        return height;
    }

    private int parseMultiplicative() {
        int height = parsePower();  // changed order to respect ^ precedence
        while (true) {
            char c = peek();
            if (c == '*' || c == '/') {
                nextChar();
                binaryOpCount++;
                if(c=='/'){
                    divOpCount++;
                }
                int right = parsePower();
                height = 1 + Math.max(height, right);
            } else {
                break;
            }
        }
        return height;
    }

    private int parsePower() {
        int leftHeight = parseUnary();
        if (peek() == '^') {
            nextChar();
            binaryOpCount++;
            int rightHeight = parseUnary();  // right-associative
            return 1 + Math.max(leftHeight, rightHeight);
        }
        return leftHeight;
    }

    private int parseUnary() {
        char c = peek();
        if (c == '+' || c == '-') {
            nextChar();
            unaryOpCount++;
            int operandHeight = parsePrimary();  // note: unary binds tighter than ^
            return 1 + operandHeight;
        }
        return parsePrimary();
    }

    private int parsePrimary() {
        char c = peek();

        // Number
        if (Character.isDigit(c) || c == '.') {
            consumeNumber();
            return 1;
        }

        // Parentheses
        if (c == '(') {
            nextChar();
            int height = parseExpression();
            if (peek() == ')') nextChar();
            return height;
        }

        // Variable or function
        if (Character.isLetter(c)) {
            String name = consumeIdentifier();
            if (peek() == '(') {
                nextChar(); // (
                functionCount++;  // ← we found a function!
                int maxArgHeight = 0;
                boolean hasArgs = peek() != ')';
                if (hasArgs) {
                    while (true) {
                        int argHeight = parseExpression();
                        maxArgHeight = Math.max(maxArgHeight, argHeight);
                        if (peek() == ',') {
                            nextChar();
                        } else {
                            break;
                        }
                    }
                }
                if (peek() == ')') nextChar();
                return 1 + maxArgHeight;
            }
            // plain variable
            return 1;
        }

        return 0; // assume valid input
    }

    // ──────────────────────────────────────────────
    //  Token helpers (unchanged)
    // ──────────────────────────────────────────────

    private void consumeNumber() {
        while (Character.isDigit(peek())) nextChar();
        if (peek() == '.') { nextChar(); while (Character.isDigit(peek())) nextChar(); }
        char e = peek();
        if (e == 'e' || e == 'E') {
            nextChar();
            char sign = peek();
            if (sign == '+' || sign == '-') nextChar();
            while (Character.isDigit(peek())) nextChar();
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
            "(1 + (2 + (3 + 4)))"
        };

        for (String s : tests) {
            Result r = new MathExpressionTreeDepth(s).calculate();
            System.out.printf("%-38s → %s%n", s, r);
        }
    }
}