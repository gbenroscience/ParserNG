package com.github.gbenroscience.parser.benchmarks;

import com.github.gbenroscience.math.Maths;
import java.util.*;

public final class TwG {

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

    public double evaluate(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            throw new IllegalArgumentException("Empty expression");
        }

        // Special case: single number
        if (tokens.size() == 1 && isNumber(tokens.get(0))) {
            return fastParseDouble(tokens.get(0));
        }

        String[] toks = tokens.toArray(new String[0]);
        int n = toks.length;

        DoubleStack values = new DoubleStack();
        IntStack operators = new IntStack();

        int i = 0;
        boolean expectOperand = true;

        while (i < n) {
            String token = toks[i];
            int len = token.length();

            // Function or operator classification
            int code = -1;
            if (len == 1) {
                char c = token.charAt(0);
                switch (c) {
                    case '+':
                        code = OP_ADD;
                        break;
                    case '-':
                        code = OP_SUB;
                        break;
                    case '*':
                        code = OP_MUL;
                        break;
                    case '/':
                        code = OP_DIV;
                        break;
                    case '%':
                        code = OP_MOD;
                        break;
                    case '^':
                        code = OP_POW;
                        break;
                    case 'Р':
                        code = OP_PERM;
                        break;
                    case 'Č':
                        code = OP_COMB;
                        break;
                    case '!':
                        code = OP_FACT;
                        break;
                    case '²':
                        code = OP_SQ;
                        break;
                    case '³':
                        code = OP_CU;
                        break;
                    case '√':
                        code = OP_SQRT;
                        break;
                    default:
                        code = -1;
                }
            } else if (len == 2) {
                if (token.charAt(0) == '-' && token.charAt(1) == '¹') {
                    code = OP_INV;
                } else if (token.charAt(0) == '³' && token.charAt(1) == '√') {
                    code = OP_CBRT;
                }
            }

            if (code != -1) {
                if (expectOperand) {
                    if (code == OP_ADD) {
                        i++;
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
                i++;
            } else {
                double val = fastParseDouble(token);
                i++;

                // Postfix folding
                while (i < n) {
                    String next = toks[i];
                    int postCode = getPostfixCode(next);
                    if (postCode == -1) {
                        break;
                    }
                    val = applyPostfix(postCode, val);
                    i++;
                }
                values.push(val);
                expectOperand = false;
            }
        }

        while (operators.size > 0) {
            applyTop(values, operators);
        }

        if (values.size != 1) {
            throw new IllegalArgumentException("Invalid expression");
        }

        return values.data[0];
    }

    private static int getPostfixCode(String s) {
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

    private double applyPostfix(int code, double x) {
        switch (code) {
            case OP_FACT:
                return Maths.fact(x);
            case OP_SQ:
                return x * x;
            case OP_CU:
                return x * x * x;
            case OP_INV:
                return 1.0 / x;
            case OP_SQRT:
                return (x < 0) ? Double.NaN : Math.sqrt(x);
            case OP_CBRT:
                return Math.cbrt(x);
            default:
                return x;
        }
    }

    private void applyTop(DoubleStack values, IntStack operators) {
        int code = operators.data[--operators.size];

        if (code == OP_UMINUS) {
            values.data[values.size - 1] = -values.data[values.size - 1];
            return;
        }

        double b = values.data[--values.size];
        double a = values.data[--values.size];
        double res;

        // Infinity / NaN safeguards
        if (Double.isNaN(a) || Double.isNaN(b)) {
            res = Double.NaN;
        } else if (Double.isInfinite(a) || Double.isInfinite(b)) {
            switch (code) {
                case OP_ADD:
                case OP_SUB:
                case OP_MUL:
                    res = Double.POSITIVE_INFINITY;
                    break;
                case OP_DIV:
                    res = (b == 0) ? Double.NaN : Double.POSITIVE_INFINITY;
                    break;
                case OP_POW:
                    res = Math.pow(a, b);
                    break;
                default:
                    res = Double.NaN;
            }
        } else {
            switch (code) {
                case OP_ADD:
                    res = a + b;
                    break;
                case OP_SUB:
                    res = a - b;
                    break;
                case OP_MUL:
                    res = a * b;
                    break;
                case OP_DIV:
                    res = a / b;
                    break;
                case OP_MOD:
                    res = a % b;
                    break;
                case OP_POW:
                    res = Math.pow(a, b);
                    break;
                case OP_PERM:
                    res = permutation(a, b);
                    break;
                case OP_COMB:
                    res = combination(a, b);
                    break;
                default:
                    res = 0.0;
            }
        }

        values.data[values.size++] = res;
    }

    double permutation(double n, double r) {
        return Maths.fact(n) / Maths.fact(n - r);
    }

    double combination(double n, double r) {
        return Maths.fact(n) / (Maths.fact(n - r) * Maths.fact(r));
    }

    // Custom fast parser (2-4x faster than Double.parseDouble for these short valid strings)
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

// Fixed-size primitive stacks (increased to 128 for extra safety)
    private class DoubleStack {

        private double[] data = new double[128];
        private int size = 0;

        void push(double v) {
            if (size == data.length) {
                data = Arrays.copyOf(data, data.length * 2);
            }
            data[size++] = v;
        }
    }

    private class IntStack {

        private int[] data = new int[128];
        private int size = 0;

        void push(int v) {
            if (size == data.length) {
                data = Arrays.copyOf(data, data.length * 2);
            }
            data[size++] = v;
        }
    }

    private static boolean isNumber(String s) {
        // Simple check used in single-number case
        return s != null && !s.isEmpty() && Character.isDigit(s.charAt(0)) || s.charAt(0) == '-' || s.charAt(0) == '+';
    }
}
