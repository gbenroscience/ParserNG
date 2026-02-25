/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.github.gbenroscience.parser.benchmarks;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathScanner;
import java.util.*;

public class TwGMBP {

    // Operator codes
    private static final int OP_ADD = 1;
    private static final int OP_SUB = 2;
    private static final int OP_MUL = 3;
    private static final int OP_DIV = 4;
    private static final int OP_MOD = 5;
    private static final int OP_POW = 6;
    private static final int OP_PERM = 7;
    private static final int OP_COMB = 8;
    private static final int OP_UMINUS = 13;
    private static final int OP_FACT = 9;
    private static final int OP_SQ = 10;
    private static final int OP_CU = 11;
    private static final int OP_INV = 12;
    private static final int OP_SQRT = 14;
    private static final int OP_CBRT = 15;

 private static int precedence(int code) {
    switch (code) {
        case OP_UMINUS:
            return 100;
        case OP_POW:
            return 90;
        case OP_MUL:
        case OP_DIV:
        case OP_MOD:
        case OP_PERM:
        case OP_COMB:
            return 80;
        case OP_ADD:
        case OP_SUB:
            return 70;
        default:
            return 0;
    }
}
    private static boolean isRightAssociative(int code) {
        return code == OP_POW || code == OP_UMINUS;
    }

    public static double evaluate(List<String> tokens) {
        String[] toks = tokens.toArray(String[]::new);
        int[] index = {0};
        double result = evaluateExpr(toks, index);
        if (index[0] != toks.length) {
            throw new IllegalArgumentException("Extra tokens after expression");
        }
        return result;
    }

    private static double evaluateExpr(String[] toks, int[] index) {
        DoubleStack values = new DoubleStack();
        IntStack operators = new IntStack();
        boolean expectOperand = true;

        while (index[0] < toks.length) {
            String token = toks[index[0]];

            if (token.equals("(")) {
                index[0]++;
                double subVal = evaluateExpr(toks, index);
                if (index[0] >= toks.length || !toks[index[0]].equals(")")) {
                    throw new IllegalArgumentException("Mismatched parentheses");
                }
                index[0]++; // consume ")"
                subVal = foldPostfix(toks, index, subVal);
                values.push(subVal);
                expectOperand = false;
                continue;
            }

            // Function call (unary or binary)
            String funcName = isFunction(token);
            if (funcName != null && expectOperand) {
                index[0]++;
                if (index[0] >= toks.length || !toks[index[0]].equals("(")) {
                    throw new IllegalArgumentException("Expected ( after function " + funcName);
                }
                index[0]++; // consume "("

                // Evaluate arguments inside ( ) — allows space-separated simple args for binary
                double[] args = evaluateFunctionArguments(toks, index);

                if (index[0] >= toks.length || !toks[index[0]].equals(")")) {
                    throw new IllegalArgumentException("Mismatched parentheses in " + funcName);
                }
                index[0]++; // consume ")"

                double result;
                if (isBinaryFunction(funcName)) {
                    if (args.length != 2) {
                        throw new IllegalArgumentException(funcName + " requires exactly 2 arguments");
                    }
                    result = applyBinaryFunction(funcName, args[0], args[1]);
                } else {
                    if (args.length != 1) {
                        throw new IllegalArgumentException(funcName + " requires exactly 1 argument");
                    }
                    result = applyUnaryFunction(funcName, args[0]);
                }

                result = foldPostfix(toks, index, result);
                values.push(result);
                expectOperand = false;
                continue;
            }

            // Operator classification
            int code = classifyOperator(token);
            if (code != -1) {
                if (expectOperand) {
                    if (code == OP_ADD) {
                        index[0]++;
                        continue;
                    }
                    if (code == OP_SUB) {
                        code = OP_UMINUS;
                    }
                }

                int prec = precedence(code);
                boolean right = isRightAssociative(code);
                while (operators.size > 0) {
                    int top = operators.data[operators.size - 1];
                    int topPrec = precedence(top);
                    if (topPrec > prec || (topPrec == prec && !right)) {
                        applyTop(values, operators);
                    } else {
                        break;
                    }
                }
                operators.push(code);
                expectOperand = true;
                index[0]++;
            } else {
                // Number
                double val = fastParseDouble(token);
                index[0]++;
                val = foldPostfix(toks, index, val);
                values.push(val);
                expectOperand = false;
            }
        }

        while (operators.size > 0) {
            applyTop(values, operators);
        }

        if (values.size != 1) {
            throw new IllegalArgumentException("Invalid expression in subexpression");
        }
        return values.data[0];
    }

    // Evaluate inside function parentheses and return array of argument values
    // Allows space-separated simple arguments for binary functions (e.g., max(3 10))
    private static double[] evaluateFunctionArguments(String[] toks, int[] index) {
        DoubleStack argValues = new DoubleStack();
        IntStack argOperators = new IntStack();
        boolean expectOperand = true;

        while (index[0] < toks.length && !toks[index[0]].equals(")")) {
            String token = toks[index[0]];

            if (token.equals("(")) {
                index[0]++;
                double sub = evaluateExpr(toks, index);
                if (index[0] >= toks.length || !toks[index[0]].equals(")")) {
                    throw new IllegalArgumentException("Mismatched parentheses in function argument");
                }
                index[0]++;
                sub = foldPostfix(toks, index, sub);
                argValues.push(sub);
                expectOperand = false;
                continue;
            }

            int code = classifyOperator(token);
            if (code != -1) {
                if (expectOperand) {
                    if (code == OP_ADD) {
                        index[0]++;
                        continue;
                    }
                    if (code == OP_SUB) {
                        code = OP_UMINUS;
                    }
                }

                int prec = precedence(code);
                boolean right = isRightAssociative(code);
                while (argOperators.size > 0) {
                    int top = argOperators.data[argOperators.size - 1];
                    int topPrec = precedence(top);
                    if (topPrec > prec || (topPrec == prec && !right)) {
                        applyTop(argValues, argOperators);
                    } else {
                        break;
                    }
                }
                argOperators.push(code);
                expectOperand = true;
                index[0]++;
            } else {
                double val = fastParseDouble(token);
                index[0]++;
                val = foldPostfix(toks, index, val);
                argValues.push(val);
                expectOperand = false;
            }
        }

        while (argOperators.size > 0) {
            applyTop(argValues, argOperators);
        }

        double[] args = new double[argValues.size];
        for (int i = 0; i < argValues.size; i++) {
            args[i] = argValues.data[i];
        }
        return args;
    }

    private static double foldPostfix(String[] toks, int[] index, double val) {
        while (index[0] < toks.length) {
            String next = toks[index[0]];
            int postCode = classifyPostfix(next);
            if (postCode == -1) break;
            val = applyPostfix(postCode, val);
            index[0]++;
        }
        return val;
    }

  private static int classifyOperator(String s) {
    int len = s.length();
    if (len == 1) {
        char c = s.charAt(0);
        switch (c) {
            case '+':
                return OP_ADD;
            case '-':
                return OP_SUB;
            case '*':
                return OP_MUL;
            case '/':
                return OP_DIV;
            case '%':
                return OP_MOD;
            case '^':
                return OP_POW;
            case 'Р':
                return OP_PERM;
            case 'Č':
                return OP_COMB;
            case '!':
                return OP_FACT;
            case '²':
                return OP_SQ;
            case '³':
                return OP_CU;
            case '√':
                return OP_SQRT;
            default:
                return -1;
        }
    } else if (len == 2) {
        if (s.charAt(0) == '-' && s.charAt(1) == '¹') {
            return OP_INV;
        }
        if (s.charAt(0) == '³' && s.charAt(1) == '√') {
            return OP_CBRT;
        }
    }
    return -1;
}

private static String isFunction(String token) {
    switch (token) {
        case "sin":
        case "cos":
        case "tan":
        case "sec":
        case "csc":
        case "cot":
        case "asin":
        case "acos":
        case "atan":
        case "asec":
        case "acsc":
        case "acot":
        case "sinh":
        case "cosh":
        case "tanh":
        case "sech":
        case "csch":
        case "coth":
        case "asinh":
        case "acosh":
        case "atanh":
        case "asech":
        case "acsch":
        case "acoth":
        case "ln":
        case "exp":
        case "fact":
        case "perm":
        case "comb":
        case "log10":
        case "pow":
        case "log":
        case "max":
        case "min":
            return token;
        default:
            return null;
    }
}





    private static boolean isBinaryFunction(String func) {
        return "pow".equals(func) || "max".equals(func) || "min".equals(func) ||
               "perm".equals(func) || "comb".equals(func) || "log".equals(func);
    }

    private static int classifyPostfix(String s) {
    int len = s.length();
    if (len == 1) {
        char c = s.charAt(0);
        switch (c) {
            case '!':
                return OP_FACT;
            case '²':
                return OP_SQ;
            case '³':
                return OP_CU;
            case '√':
                return OP_SQRT;
            default:
                return -1;
        }
    } else if (len == 2) {
        if (s.charAt(0) == '-' && s.charAt(1) == '¹') {
            return OP_INV;
        }
        if (s.charAt(0) == '³' && s.charAt(1) == '√') {
            return OP_CBRT;
        }
    }
    return -1;
}

private static double applyUnaryFunction(String func, double x) {
    if (func == null) {
        throw new IllegalArgumentException("Unknown unary function: null");
    }

    switch (func) {
        case "sin":     return Math.sin(x);
        case "cos":     return Math.cos(x);
        case "tan":     return Math.tan(x);
        case "sec":     return 1.0 / Math.cos(x);
        case "csc":     return 1.0 / Math.sin(x);
        case "cot":     return 1.0 / Math.tan(x);
        case "asin":    return Math.asin(x);
        case "acos":    return Math.acos(x);
        case "atan":    return Math.atan(x);
        case "asec":    return Math.acos(1.0 / x);
        case "acsc":    return Math.asin(1.0 / x);
        case "acot":    return Math.atan(1.0 / x);
        case "sinh":    return Math.sinh(x);
        case "cosh":    return Math.cosh(x);
        case "tanh":    return Math.tanh(x);
        case "sech":    return 1.0 / Math.cosh(x);
        case "csch":    return 1.0 / Math.sinh(x);
        case "coth":    return 1.0 / Math.tanh(x);
        case "asinh":   return Maths.asinh(x);
        case "acosh":   return Maths.acosh(x);
        case "atanh":   return Maths.atanh(x);
        case "asech":   return Maths.asech(x);
        case "acsch":   return Maths.acsch(x);
        case "acoth":   return Maths.acoth(x);
        case "ln":      return Maths.log(x);
        case "exp":     return Maths.exp(x);
        case "fact":    return Maths.fact(x);
        case "log10":   return Maths.logToAnyBase(x, 10);
        default:
            throw new IllegalArgumentException("Unknown unary function: " + func);
    }
}

private static double applyBinaryFunction(String func, double a, double b) {
    if (func == null) {
        throw new IllegalArgumentException("Unknown binary function: null");
    }

    switch (func) {
        case "pow":     return Math.pow(a, b);
        case "max":     return Math.max(a, b);
        case "min":     return Math.min(a, b);
        case "log":     return Maths.logToAnyBase(a, b);
        case "perm":    return permutation(a, b);
        case "comb":    return combination(a, b);
        default:
            throw new IllegalArgumentException("Unknown binary function: " + func);
    }
}

private static double applyPostfix(int code, double x) {
    switch (code) {
        case OP_FACT:   return realFactorial(x);
        case OP_SQ:     return x * x;
        case OP_CU:     return x * x * x;
        case OP_INV:    return 1.0 / x;
        case OP_SQRT:   return Math.sqrt(x);
        case OP_CBRT:   return Math.cbrt(x);
        default:
            throw new IllegalArgumentException("Invalid postfix: " + code);
    }
}

private static void applyTop(DoubleStack values, IntStack operators) {
    int code = operators.data[--operators.size];

    if (code == OP_UMINUS) {
        values.data[values.size - 1] = -values.data[values.size - 1];
        return;
    }

    double b = values.data[--values.size];
    double a = values.data[--values.size];
    double res;

    switch (code) {
        case OP_ADD:    res = a + b; break;
        case OP_SUB:    res = a - b; break;
        case OP_MUL:    res = a * b; break;
        case OP_DIV:    res = a / b; break;
        case OP_MOD:    res = a % b; break;
        case OP_POW:    res = Math.pow(a, b); break;
        case OP_PERM:   res = Maths.fact(a) / Maths.fact(a - b); break;
        case OP_COMB:   res = Maths.fact(a) / (Maths.fact(b) * Maths.fact(a - b)); break;
        default:
            throw new IllegalArgumentException("Invalid operator: " + code);
    }

    values.data[values.size++] = res;
}

    // Your exact ancient Stirling-based real factorial (cleaned)
    private static double realFactorial(double p) {
        double prod = 1.0;
        double fact;
        double dbVal = p;
        double n = Math.floor(dbVal);
        double k = dbVal - n;
        double d = 160 + k;
        if (dbVal == 0 || dbVal == 1.0) {
            fact = 1.0;
        } else if (dbVal < 0 && k == 0) {
            fact = Double.NEGATIVE_INFINITY;
        } else if (dbVal <= d && k == 0) {
            double temp = dbVal;
            while (temp > 0) {
                prod *= temp;
                temp--;
            }
            fact = prod;
        } else if (dbVal <= d && k != 0) {
            double d2 = d * d;
            double d3 = d2 * d;
            double d4 = d3 * d;
            double d5 = d4 * d;
            double d6 = d5 * d;
            double d7 = d6 * d;
            double d8 = d7 * d;
            double d9 = d8 * d;
            double fact1 = Math.pow(d / Math.E, d) * Math.sqrt(2 * d * Math.PI) *
                    (1 + 1/(12*d) + 1/(288*d2) - 139/(51840*d3) -
                     59909/(2.592E8*d4) + 1208137/(1.492992E9*d5) -
                     1151957/(1.875E11*d6) - 101971/(2.88E8*d7) +
                     189401873/(2.4E12*d8) + 1293019/(1.8E10*d9));
            double i = 1;
            while (n + i <= 160) {
                prod *= (dbVal + i);
                i++;
            }
            fact = fact1 / prod;
        } else {
            double v2 = dbVal * dbVal;
            double v3 = v2 * dbVal;
            double v4 = v3 * dbVal;
            double v5 = v4 * dbVal;
            double v6 = v5 * dbVal;
            double v7 = v6 * dbVal;
            double v8 = v7 * dbVal;
            double v9 = v8 * dbVal;
            fact = Math.pow(dbVal / Math.E, dbVal) * Math.sqrt(2 * dbVal * Math.PI) *
                    (1 + 1/(12*dbVal) + 1/(288*v2) - 139/(51840*v3) -
                     59909/(2.592E8*v4) + 1208137/(1.492992E9*v5) -
                     1151957/(1.875E11*v6) - 101971/(2.88E8*v7) +
                     189401873/(2.4E12*v8) + 1293019/(1.8E10*v9));
        }
        return fact;
    }

    // Custom fast parser
    private static double fastParseDouble(String s) {
        int len = s.length();
        int idx = 0;
        double value = 0.0;
        while (idx < len) {
            char c = s.charAt(idx);
            if (c >= '0' && c <= '9') {
                value = value * 10.0 + (c - '0');
                idx++;
            } else {
                break;
            }
        }
        if (idx < len && s.charAt(idx) == '.') {
            idx++;
            double frac = 0.0;
            double div = 1.0;
            while (idx < len) {
                char c = s.charAt(idx);
                if (c >= '0' && c <= '9') {
                    div *= 10.0;
                    frac = frac * 10.0 + (c - '0');
                    idx++;
                } else {
                    break;
                }
            }
            value += frac / div;
        }
        if (idx < len && (s.charAt(idx) == 'e' || s.charAt(idx) == 'E')) {
            idx++;
            int exp = 0;
            boolean negExp = false;
            if (idx < len && s.charAt(idx) == '-') {
                negExp = true;
                idx++;
            } else if (idx < len && s.charAt(idx) == '+') {
                idx++;
            }
            while (idx < len) {
                char c = s.charAt(idx);
                exp = exp * 10 + (c - '0');
                idx++;
            }
            value *= Math.pow(10.0, negExp ? -exp : exp);
        }
        return value;
    }

    private static double permutation(double n, double r) {
        return Maths.fact(n) / Maths.fact(n - r);
    }

    private static double combination(double n, double r) {
        return Maths.fact(n) / (Maths.fact(n - r) * Maths.fact(r));
    }

    private static class DoubleStack {
        private final double[] data = new double[64];
        private int size = 0;
        void push(double v) { data[size++] = v; }
    }

    private static class IntStack {
        private final int[] data = new int[64];
        private int size = 0;
        void push(int v) { data[size++] = v; }
    }
    
    public static void main(String[] args) {
         String s1= "sin(12)";
             MathExpression me = new MathExpression(s1);
         System.out.println("me.solve(): "+me.solve()); 
         System.out.println("ms.scanner(): "+me.getScanner());
         double v = evaluate(new MathScanner(s1).scanner());
         System.out.println("v = "+v);
    }
}
        