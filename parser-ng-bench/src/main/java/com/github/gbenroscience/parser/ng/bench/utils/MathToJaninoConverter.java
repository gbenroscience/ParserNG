package com.github.gbenroscience.parser.ng.bench.utils;


 

import java.util.*;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 *
 * @author GBEMIRO
 */
public class MathToJaninoConverter {

    private static final Set<String> MATH_FUNCS = new HashSet<>(Arrays.asList(
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "exp", "log", "log10", "sqrt", "cbrt", "abs", "ceil", "floor", "round"
    ));

    public static String convert(String expr) {
        if (expr == null) return null;
        
        // 1. Basic replacements
        String result = expr.replaceAll("(?i)\\bpi\\b", "Math.PI")
                            .replaceAll("(?i)\\be\\b", "Math.E");

        // 2. Handle Factorial first (it's postfix and simpler)
        result = processPostfix(result, '!', "factorial");

        // 3. Handle Power (The tricky one)
        result = processPower(result);

        // 4. Wrap standard functions
        for (String func : MATH_FUNCS) {
            result = result.replaceAll("(?<!\\.)\\b" + func + "\\s*\\(", "Math." + func + "(");
        }

        return result;
    }

    private static String processPower(String expr) {
        while (expr.contains("^")) {
            int opIdx = expr.indexOf('^');
            
            // Find Left Operand
            int leftStart = findLeftOperand(expr, opIdx);
            String left = expr.substring(leftStart, opIdx).trim();
            
            // Find Right Operand
            int rightEnd = findRightOperand(expr, opIdx);
            String right = expr.substring(opIdx + 1, rightEnd).trim();
            
            String replacement = "Math.pow(" + left + ", " + right + ")";
            expr = expr.substring(0, leftStart) + replacement + expr.substring(rightEnd);
        }
        return expr;
    }

    private static int findLeftOperand(String s, int opIdx) {
        int i = opIdx - 1;
        int parenDepth = 0;
        // Skip trailing whitespace
        while (i >= 0 && Character.isWhitespace(s.charAt(i))) i--;
        
        while (i >= 0) {
            char c = s.charAt(i);
            if (c == ')') parenDepth++;
            else if (c == '(') parenDepth--;
            
            if (parenDepth == 0 && (isBoundary(c) || Character.isWhitespace(c))) {
                // Check if this was a function name (e.g. sin()^2)
                int temp = i;
                while (temp >= 0 && Character.isLetterOrDigit(s.charAt(temp))) temp--;
                return temp + 1;
            }
            if (parenDepth < 0) return i + 1;
            i--;
        }
        return 0;
    }

    private static int findRightOperand(String s, int opIdx) {
        int i = opIdx + 1;
        int parenDepth = 0;
        int len = s.length();
        // Skip leading whitespace
        while (i < len && Character.isWhitespace(s.charAt(i))) i++;
        
        // Handle unary minus: 2^-x
        if (i < len && (s.charAt(i) == '-' || s.charAt(i) == '+')) i++;

        while (i < len) {
            char c = s.charAt(i);
            if (c == '(') parenDepth++;
            else if (c == ')') parenDepth--;
            
            if (parenDepth == 0 && (isBoundary(c) || Character.isWhitespace(c))) return i;
            if (parenDepth < 0) return i;
            i++;
        }
        return len;
    }

    private static String processPostfix(String expr, char op, String funcName) {
        while (expr.indexOf(op) != -1) {
            int idx = expr.indexOf(op);
            int start = findLeftOperand(expr, idx);
            String operand = expr.substring(start, idx).trim();
            expr = expr.substring(0, start) + funcName + "(" + operand + ")" + expr.substring(idx + 1);
        }
        return expr;
    }

    private static boolean isBoundary(char c) {
        return c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == ',';
    }

    public static void main(String[] args) {
        String[] inputs = {
            "(x^2+y^0.5)^4.2",
            "((x^2 + 3*sin(x+5^3-1/4)) / (23/33 + cos(x^2))) * (exp(x) / 10) + (sin(3) + cos(4 - sin(2))) ^ (-2)",
            "sin(x^3+y^3)-4*(x-y)",
            "(x + 5)!",
            "sin(x)^2 + cos(y)^2"
        };

        for (String in : inputs) {
            System.out.println("Original: " + in);
            System.out.println("Janino:   " + convert(in));
            System.out.println("---");
        }
    }
}