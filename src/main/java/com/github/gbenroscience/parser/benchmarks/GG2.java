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


 
public class GG2 {
 public double evaluate(List<String> toks) {
    int len = toks.size();
      final String[] tokens = toks.toArray(new String[0]);
    double[] valStack = new double[len];
    char[] opStack = new char[len];
    int vPtr = 0;
    int oPtr = 0;

    for (int i = 0; i < len; i++) {
        String s = tokens[i];
        if (s == null || s.isEmpty()) continue;
        
        char c0 = s.charAt(0);

        // 1. IMPROVED NUMERIC CHECK
        // Only parse as number if it's a digit OR a minus followed by a digit
        if ((c0 >= '0' && c0 <= '9') || (s.length() > 1 && c0 == '-' && s.charAt(1) >= '0' && s.charAt(1) <= '9')) {
            valStack[vPtr++] = fastParseDouble(s);
        } 
        // 2. GROUPS
        else if (c0 == '(') {
            opStack[oPtr++] = '(';
        } 
        else if (c0 == ')') {
            while (oPtr > 0 && opStack[oPtr - 1] != '(') {
                vPtr = applyOp(opStack[--oPtr], valStack, vPtr);
            }
            if (oPtr > 0) oPtr--; 
        } 
        // 3. OPERATORS & POSTFIX
        else {
            int p = getPrec(s);
            while (oPtr > 0 && opStack[oPtr - 1] != '(' && getPrec(String.valueOf(opStack[oPtr - 1])) >= p) {
                // Right-associativity check for power
                if (c0 == '^' && opStack[oPtr - 1] == '^') break;
                vPtr = applyOp(opStack[--oPtr], valStack, vPtr);
            }
            
            // Map special tokens to internal chars
            if (s.equals("³√")) opStack[oPtr++] = 'R';
            else if (s.equals("-¹")) opStack[oPtr++] = 'I'; // 'I' for Inverse
            else opStack[oPtr++] = c0;
        }
    }

    while (oPtr > 0) {
        vPtr = applyOp(opStack[--oPtr], valStack, vPtr);
    }
    return valStack[0];
}

private static int getPrec(String op) {
    char c = op.charAt(0);
    // Treat "-¹" (Inverse) with high postfix precedence
    if (op.equals("-¹")) return 4; 
    
    switch (c) {
        case '+': case '-': return 1;
        case '*': case '/': case '%': case 'Р': case 'Č': return 2;
        case '^': return 3;
        case '!': case '²': case '³': case '√': case 'R': case 'I': return 4;
        default: return 0;
    }
}

private static int applyOp(char op, double[] stack, int ptr) {
    // Postfix/Unary (One operand)
    if ("!²³√RI".indexOf(op) != -1) {
        double b = stack[--ptr];
        switch (op) {
            case '!': stack[ptr++] = Maths.fact(b); break;
            case '²': stack[ptr++] = b * b; break;
            case '³': stack[ptr++] = b * b * b; break;
            case '√': stack[ptr++] = Math.sqrt(b); break;
            case 'R': stack[ptr++] = Math.cbrt(b); break;
            case 'I': stack[ptr++] = 1.0 / b; break;
        }
        return ptr;
    }

    // Binary (Two operands)
    double b = stack[--ptr];
    double a = stack[--ptr];
    switch (op) {
        case '+': stack[ptr++] = a + b; break;
        case '-': stack[ptr++] = a - b; break;
        case '*': stack[ptr++] = a * b; break;
        case '/': stack[ptr++] = a / b; break;
        case '%': stack[ptr++] = a % b; break;
        case '^': stack[ptr++] = Math.pow(a, b); break;
        case 'Р': stack[ptr++] = Maths.fact(a) / Maths.fact(a - b); break;
        case 'Č': stack[ptr++] = Maths.fact(a) / (Maths.fact(b) * Maths.fact(a - b)); break;
    }
    return ptr;
}

    private static double fastParseDouble(String s) {
        // ... Your existing fastParseDouble ...
        // (Ensuring it handles scientific notation correctly)
        return Double.parseDouble(s); // Use native for a second to verify accuracy first
    }
}

/*
public class GG2 {

 
    public double evaluate(List<String> toks) {
        String[] tokens = toks.toArray(new String[0]);
        int len = tokens.length;
        // Use local primitive arrays as stacks (Pre-allocated based on token count)
        double[] valStack = new double[len];
        int[] opStack = new int[len];
        int valPtr = 0;
        int opPtr = 0;

        for (int i = 0; i < len; i++) {
            String token = tokens[i];
            if (token == null || token.isEmpty()) {
                continue;
            }

            char first = token.charAt(0);

            // 1. FAST NUMBER PARSING
            // Checks for digits or a leading negative sign for a number
            if ((first >= '0' && first <= '9') || (token.length() > 1 && first == '-')) {
                valStack[valPtr++] = fastParseDouble(token);
            } // 2. POSTFIX OPERATORS (Applied to the value already on the stack)
            else if (token.equals("!") || token.equals("²") || token.equals("³")
                    || token.equals("-¹") || token.equals("√") || token.equals("³√")) {
                if (valPtr > 0) {
                    applyPostfix(valStack, valPtr - 1, token);
                }
            } // 3. GROUPING
            else if (first == '(') {
                opStack[opPtr++] = '(';
            } else if (first == ')') {
                while (opPtr > 0 && opStack[opPtr - 1] != '(') {
                    valPtr = applyBinary(valStack, valPtr, (char) opStack[--opPtr]);
                }
                opPtr--; // Discard the '('
            } // 4. BINARY OPERATORS (Shunting-Yard)
            else {
                int prec = getPrecedence(first);
                while (opPtr > 0 && opStack[opPtr - 1] != '(' && getPrecedence((char) opStack[opPtr - 1]) >= prec) {
                    valPtr = applyBinary(valStack, valPtr, (char) opStack[--opPtr]);
                }
                opStack[opPtr++] = first;
            }
        }

        // Finalize remaining operations
        while (opPtr > 0) {
            valPtr = applyBinary(valStack, valPtr, (char) opStack[--opPtr]);
        }

        return valStack[0];
    }

    private static void applyPostfix(double[] stack, int idx, String op) {
        double v = stack[idx];
        switch (op) {
            case "!":
                stack[idx] = Maths.fact(v);
                break;
            case "²":
                stack[idx] = v * v;
                break;
            case "³":
                stack[idx] = v * v * v;
                break;
            case "-¹":
                stack[idx] = 1.0 / v;
                break;
            case "√":
                stack[idx] = Math.sqrt(v);
                break;
            case "³√":
                stack[idx] = Math.cbrt(v);
                break;
        }
    }

    private static int applyBinary(double[] stack, int ptr, char op) {
        if (ptr < 2) {
            return ptr;
        }
        double b = stack[--ptr];
        double a = stack[--ptr];
        switch (op) {
            case '+':
                stack[ptr++] = a + b;
                break;
            case '-':
                stack[ptr++] = a - b;
                break;
            case '*':
                stack[ptr++] = a * b;
                break;
            case '/':
                stack[ptr++] = a / b;
                break;
            case '%':
                stack[ptr++] = a % b;
                break;
            case '^':
                stack[ptr++] = Math.pow(a, b);
                break;
            case 'Р':
                stack[ptr++] = nPr(a, b);
                break;
            case 'Č':
                stack[ptr++] = nCr(a, b);
                break;
        }
        return ptr;
    }

    private int getPrecedence(char op) {
        switch (op) {
            case '+':
            case '-':
                return 1;
            case '*':
            case '/':
            case '%':
                return 2;
            case '^':
            case 'Р':
            case 'Č':
                return 3;
            case '√':
            case '³':
            case '²': 
                return 4;
            default:
                return 0;
        }
    }

    // --- High Performance Utilities ---
    private double fastParseDouble(String s) {
        int len = s.length();
        if (len == 0) {
            return 0.0; // or throw if empty not allowed
        }
        int idx = 0;
        boolean negative = false;
        char first = s.charAt(0);

        // Handle sign
        if (first == '-') {
            negative = true;
            idx = 1;
        } else if (first == '+') {
            idx = 1;
        }

        double value = 0.0;

        // Integer part
        boolean hasIntPart = false;
        while (idx < len) {
            char c = s.charAt(idx);
            if (c >= '0' && c <= '9') {
                value = value * 10.0 + (c - '0');
                hasIntPart = true;
                idx++;
            } else {
                break;
            }
        }

        // Fractional part
        boolean hasFracPart = false;
        if (idx < len && s.charAt(idx) == '.') {
            idx++;
            double frac = 0.0;
            double div = 1.0;
            while (idx < len) {
                char c = s.charAt(idx);
                if (c >= '0' && c <= '9') {
                    div *= 10.0;
                    frac = frac * 10.0 + (c - '0');
                    hasFracPart = true;
                    idx++;
                } else {
                    break;
                }
            }
            value += frac / div;
        }

        // Scientific notation
        if (idx < len && (s.charAt(idx) == 'e' || s.charAt(idx) == 'E')) {
            idx++;
            boolean negExp = false;
            if (idx < len && s.charAt(idx) == '-') {
                negExp = true;
                idx++;
            } else if (idx < len && s.charAt(idx) == '+') {
                idx++;
            }

            long exp = 0;
            boolean hasExpDigits = false;
            while (idx < len) {
                char c = s.charAt(idx);
                if (c >= '0' && c <= '9') {
                    exp = exp * 10 + (c - '0');
                    hasExpDigits = true;
                    if (exp > 1000000000L) { // Prevent insane overflow
                        value = negative ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
                        break;
                    }
                    idx++;
                } else {
                    break;
                }
            }
            if (hasExpDigits) {
                value *= Math.pow(10.0, negExp ? -exp : exp);
            }
        }

        // Basic validation (since scanner is trusted, but good to have)
        if (idx != len || (!hasIntPart && !hasFracPart)) {
            return 0.0; // or throw if strict
        }

        return negative ? -value : value;
    }

    private static double nPr(double n, double r) {
        if (n < r) {
            return 0;
        }
        return Maths.fact(n) / Maths.fact(n - r);
    }

    private static double nCr(double n, double r) {
        if (n < r) {
            return 0;
        }
        return Maths.fact(n) / (Maths.fact(r) * Maths.fact(n - r));
    }
}
*/