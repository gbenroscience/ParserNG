package com.github.gbenroscience.parser.benchmarks;

import java.util.*;
import java.math.BigInteger;

public class MC {

    private static final Map<Integer, BigInteger> FACT_CACHE = new HashMap<>();

    static {
        FACT_CACHE.put(0, BigInteger.ONE);
        FACT_CACHE.put(1, BigInteger.ONE);
    }

    private static BigInteger factorial(int n) {
        if (FACT_CACHE.containsKey(n)) {
            return FACT_CACHE.get(n);
        }
        BigInteger result = FACT_CACHE.get(FACT_CACHE.size() - 1);
        for (int i = FACT_CACHE.size(); i <= n; i++) {
            result = result.multiply(BigInteger.valueOf(i));
            FACT_CACHE.put(i, result);
        }
        return result;
    }

    private static BigInteger permutation(int n, int k) {
        BigInteger result = BigInteger.ONE;
        for (int i = 0; i < k; i++) {
            result = result.multiply(BigInteger.valueOf(n - i));
        }
        return result;
    }

    private static BigInteger combination(int n, int k) {
        if (k > n) {
            return BigInteger.ZERO;
        }
        return permutation(n, k).divide(factorial(k));
    }

    public static double evaluate(List<String> tokens) {
        int n = tokens.size();
        double[] valStack = new double[n];
        String[] opStack = new String[n];
        int vIdx = -1;
        int oIdx = -1;

        for (String s : tokens) {
            char c0 = s.charAt(0);

            // 1. Numeric parsing
            if ((c0 >= '0' && c0 <= '9') || (c0 == '-' && s.length() > 1 && Character.isDigit(s.charAt(1)))) {
                valStack[++vIdx] = fastParseDouble(s);
                continue;
            }

            // 2. Unary operators
            if (isUnaryOperator(s)) {
                valStack[vIdx] = applyUnary(s, valStack[vIdx]);
                continue;
            }

            // 3. Binary operators with precedence
            int currentPrec = precedence(s);
            while (oIdx >= 0) {
                int topPrec = precedence(opStack[oIdx]);
                if (s.equals("^") ? topPrec > currentPrec : topPrec >= currentPrec) {
                    double b = valStack[vIdx--];
                    double a = valStack[vIdx--];
                    valStack[++vIdx] = applyBinary(opStack[oIdx--], a, b);
                } else {
                    break;
                }
            }
            opStack[++oIdx] = s;
        }

        while (oIdx >= 0) {
            double b = valStack[vIdx--];
            double a = valStack[vIdx--];
            valStack[++vIdx] = applyBinary(opStack[oIdx--], a, b);
        }
        return valStack[0];
    }

    private static boolean isUnaryOperator(String s) {
        return s.equals("!") || s.equals("²") || s.equals("³") || s.equals("√") || s.equals("³√") || s.equals("-¹");
    }

    private static double applyUnary(String token, double a) {
        switch (token) {
            case "!":
                return factorial((int) a).doubleValue();
            case "²":
                return a * a;
            case "³":
                return a * a * a;
            case "√":
                return Math.sqrt(a);
            case "³√":
                return Math.cbrt(a);
            case "-¹":
                return 1.0 / a;
            default:
                return a;
        }
    }

    private static int precedence(String op) {
        switch (op) {
            case "!":
            case "²":
            case "³":
            case "√":
            case "³√":
            case "-¹":
                return 4;
            case "^":
                return 3;
            case "*":
            case "/":
            case "%":
            case "Р":
            case "Č":
                return 2;
            case "+":
            case "-":
                return 1;
            default:
                return 0;
        }
    }

    private static double fastParseDouble(String s) {
        try {
            if (s.indexOf('e') != -1 || s.indexOf('E') != -1) {
                return Double.parseDouble(s);
            }
            double res = 0;
            int i = 0, len = s.length();
            boolean neg = false;
            if (s.charAt(0) == '-') {
                neg = true;
                i++;
            }
            while (i < len && s.charAt(i) != '.') {
                res = res * 10 + (s.charAt(i++) - '0');
            }
            if (i < len && s.charAt(i) == '.') {
                i++;
                double div = 10;
                while (i < len) {
                    res += (s.charAt(i++) - '0') / div;
                    div *= 10;
                }
            }
            return neg ? -res : res;
        } catch (Exception e) {
            return Double.parseDouble(s);
        }
    }

    private static double applyBinary(String op, double a, double b) {
        if (op == null) {
            return 0.0;
        }

        switch (op) {
            case "+":
                return a + b;
            case "-":
                return a - b;
            case "*":
                return a * b;
            case "/":
                return a / b;
            case "%":
                return a % b;
            case "^":
                return Math.pow(a, b);
            case "Р":
                return permutation((int) a, (int) b).doubleValue();
            case "Č":
                return combination((int) a, (int) b).doubleValue();
            default:
                return 0.0;
        }
    }

    // Example usage
    public static void main(String[] args) {
        List<String> tokens1 = Arrays.asList("9", "√");
        System.out.println("√9 = " + evaluate(tokens1)); // 3.0

        List<String> tokens2 = Arrays.asList("27", "³√");
        System.out.println("³√27 = " + evaluate(tokens2)); // 3.0

        List<String> tokens3 = Arrays.asList("5", "³");
        System.out.println("5³ = " + evaluate(tokens3)); // 125.0
    }
}
