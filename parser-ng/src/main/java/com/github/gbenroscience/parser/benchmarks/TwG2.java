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
package com.github.gbenroscience.parser.benchmarks;

import com.github.gbenroscience.math.Maths;
import java.util.List;
 
 

public final class TwG2 {

    public static double evaluate(List<String> toks) {
        if (toks == null || toks.isEmpty()) {
            throw new IllegalArgumentException("Invalid expression");
        }

        int len = toks.size();
        double[] valStack = new double[len + 1];
        char[] opStack = new char[len + 1];
        int valPtr = 0;
        int opPtr = 0;

        for (int i = 0; i < len; i++) {
            String token = toks.get(i);
            if (token == null || token.isEmpty()) continue;

            char first = token.charAt(0);
            
            // --- 1. THE "STACK MISMATCH" FIX: Implicit Multiplication ---
            // If the current token is a number or '(', and the previous was a number, ')', or postfix
            if (i > 0 && needsImplicitMul(toks.get(i - 1), token)) {
                // Manually process a '*' with its correct precedence (2)
                while (opPtr > 0 && opStack[opPtr - 1] != '(' && getPrecedence('*') <= getPrecedence(opStack[opPtr - 1])) {
                    valPtr = applyBinary(valStack, valPtr, opStack[--opPtr]);
                }
                opStack[opPtr++] = '*';
            }

            // --- 2. MAIN PARSING LOGIC ---
            if (isNumber(token)) {
                valStack[valPtr++] = Double.parseDouble(token);
            } 
            else if (first == '(') {
                opStack[opPtr++] = '(';
            } 
            else if (first == ')') {
                while (opPtr > 0 && opStack[opPtr - 1] != '(') {
                    valPtr = applyBinary(valStack, valPtr, opStack[--opPtr]);
                }
                if (opPtr > 0) opPtr--; // Pop the '('
            } 
            else if (isPostfix(token)) {
                if (valPtr == 0) throw new IllegalArgumentException("Postfix missing operand");
                applyPostfix(valStack, valPtr - 1, token);
            } 
            else {
                // Unary Minus Check
                if (first == '-' && (i == 0 || isOperator(toks.get(i - 1)) || toks.get(i - 1).equals("("))) {
                    valStack[valPtr++] = 0.0; // 0 - x trick
                }

                int prec = getPrecedence(first);
                while (opPtr > 0 && opStack[opPtr - 1] != '(' && getPrecedence(opStack[opPtr - 1]) >= prec) {
                    // Power operator '^' is right-associative
                    if (first == '^' && opStack[opPtr - 1] == '^') break; 
                    valPtr = applyBinary(valStack, valPtr, opStack[--opPtr]);
                }
                opStack[opPtr++] = first;
            }
        }

        // Final Flush
        while (opPtr > 0) {
            valPtr = applyBinary(valStack, valPtr, opStack[--opPtr]);
        }

        if (valPtr != 1) {
            throw new IllegalArgumentException("Invalid expression: Stack mismatch (" + valPtr + " items left)");
        }
        return valStack[0];
    }

    private static boolean needsImplicitMul(String prev, String curr) {
        char p = prev.charAt(0);
        char c = curr.charAt(0);
        boolean pIsVal = isNumber(prev) || p == ')' || isPostfix(prev);
        boolean cIsVal = isNumber(curr) || c == '(';
        return pIsVal && cIsVal;
    }

    private static boolean isNumber(String s) {
        int len = s.length();
        char c = s.charAt(0);
        char d = len > 1 ? s.charAt(1) : '~';
        return ( (c >= '0' && c <= '9') || c == '.' || (len > 1 && c == '-' && d != '¹') ) ;
    }

    private static boolean isPostfix(String s) {
        char c = s.charAt(0);
        return (s.length() == 1 && (c == '!' || c == '²' || c == '³' || c == '√')) || s.equals("-¹") || s.equals("³√");
    }

    private static boolean isOperator(String s) {
        if (s.length() != 1) return false;
        return "+-*/%^РČ".indexOf(s.charAt(0)) != -1;
    }

    private static int getPrecedence(char op) {
        switch (op) {
            case '+': case '-': return 1;
            case '*': case '/': case '%': return 2;
            case '^': case 'Р': case 'Č': return 3;
            default: return 0;
        }
    }

    private static int applyBinary(double[] stack, int ptr, char op) {
        if (ptr < 2) throw new IllegalArgumentException("Missing operand for " + op);
        double b = stack[--ptr];
        double a = stack[--ptr];
        switch (op) {
            case '+': stack[ptr++] = a + b; break;
            case '-': stack[ptr++] = a - b; break;
            case '*': stack[ptr++] = a * b; break;
            case '/': stack[ptr++] = a / b; break;
            case '%': stack[ptr++] = a % b; break;
            case '^': stack[ptr++] = Math.pow(a, b); break;
            case 'Р': stack[ptr++] = nPr(a, b); break;
            case 'Č': stack[ptr++] = nCr(a, b); break;
        }
        return ptr;
    }

    private static void applyPostfix(double[] stack, int idx, String token) {
        double v = stack[idx];
        switch (token) {
            case "!":  stack[idx] = Maths.fact(v); break;
            case "²":  stack[idx] = v * v; break;
            case "³":  stack[idx] = v * v * v; break;
            case "-¹": stack[idx] = 1.0 / v; break;
            case "√":  stack[idx] = Math.sqrt(v); break;
            case "³√": stack[idx] = Math.cbrt(v); break;
        }
    }

    // --- Math Helpers ---
 

    private static double nPr(double n, double r) {
        if (n < r) return 0;
        return Maths.fact(n) / Maths.fact(n - r);
    }

    private static double nCr(double n, double r) {
        if (n < r) return 0;
        return Maths.fact(n) / (Maths.fact(r) * Maths.fact(n - r));
    }
}