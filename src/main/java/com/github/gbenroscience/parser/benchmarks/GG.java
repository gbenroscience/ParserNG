package com.github.gbenroscience.parser.benchmarks;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.math.Maths;
import java.util.*;

public class GG {

  // List-based wrapper for compatibility
    public static double evaluate(List<String> tokens) {
        return evaluate(tokens.toArray(new String[0]));
    }
    /**
     * Array-based entry point for maximum performance.
     */
    public static double evaluate(String[] ts) {
        final int n = ts.length;
        final double[] valStack = new double[n];
        final int[] opStack = new int[n];
        int vIdx = -1;
        int oIdx = -1;

        for (int i = 0; i < n; i++) {
            final String s = ts[i];
            if (s == null || s.isEmpty()) continue;

            final char c0 = s.charAt(0);
            final int sLen = s.length();

            // 1. NUMERIC CHECK 
            // Ensures -5 is a number, but -¹ and - are not.
            if ((c0 >= '0' && c0 <= '9') || (c0 == '-' && sLen > 1 && s.charAt(1) >= '0' && s.charAt(1) <= '9')) {
                valStack[++vIdx] = fastParseDouble(s);
                continue;
            }

            // 2. UNARY PREFIX (ROOTS)
            if (c0 == '√' || s.equals("³√")) {
                opStack[++oIdx] = (c0 == '√') ? '√' : 'R'; 
                continue;
            }

            // 3. UNARY POSTFIX (FACTORIAL/POWERS/INVERSE)
            // Explicitly handles the "-¹" token
            if (c0 == '!' || c0 == '²' || c0 == '³' || s.equals("-¹")) {
                final double a = valStack[vIdx];
                if (c0 == '!') valStack[vIdx] = Maths.fact(a);
                else if (c0 == '²') valStack[vIdx] = a * a;
                else if (c0 == '³') valStack[vIdx] = a * a * a;
                else if (s.equals("-¹")) valStack[vIdx] = 1.0 / a; // Fixed: Explicit inverse
                continue;
            }

            // 4. BINARY OPERATORS
            final int currentPrec = getPrec(c0);
            while (oIdx >= 0) {
                final char topOp = (char) opStack[oIdx];
                final int topPrec = getPrec(topOp);

                if (c0 == '^' ? topPrec > currentPrec : topPrec >= currentPrec) {
                    vIdx = applyOp((char) opStack[oIdx--], valStack, vIdx);
                } else {
                    break;
                }
            }
            opStack[++oIdx] = c0;
        }

        while (oIdx >= 0) {
            vIdx = applyOp((char) opStack[oIdx--], valStack, vIdx);
        }
        return valStack[0];
    }

    private static int getPrec(char op) {
        switch (op) {
            case '+': case '-': return 1;
            case '*': case '/': case '%': case 'Р': case 'Č': return 2;
            case '^': return 3;
            case '√': case 'R': return 4;
            default: return 0;
        }
    }

    private static int applyOp(char op, double[] valStack, int vIdx) {
        if (op == '√') {
            valStack[vIdx] = Math.sqrt(valStack[vIdx]);
            return vIdx;
        }
        if (op == 'R') {
            valStack[vIdx] = Math.cbrt(valStack[vIdx]);
            return vIdx;
        }
        
        double b = valStack[vIdx--];
        double a = valStack[vIdx];
        switch (op) {
            case '+': valStack[vIdx] = a + b; break;
            case '-': valStack[vIdx] = a - b; break;
            case '*': valStack[vIdx] = a * b; break;
            case '/': valStack[vIdx] = a / b; break;
            case '%': valStack[vIdx] = a % b; break;
            case '^': valStack[vIdx] = Math.pow(a, b); break;
            case 'Р': valStack[vIdx] = Maths.fact(a) / Maths.fact(a - b); break;
            case 'Č': valStack[vIdx] = Maths.fact(a) / (Maths.fact(b) * Maths.fact(a - b)); break;
        }
        return vIdx;
    }
    
    
      private static double fastParseDouble(String s) {
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

   
}

 