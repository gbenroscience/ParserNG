package com.github.gbenroscience.math.differentialcalculus.symbolic;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;

import java.util.HashMap;
import java.util.Map;

/**
 * GC-Free Forward Mode Symbolic Differentiation Evaluator. Refactored to use a
 * decoupled rule registry for extensibility.
 */
public class SymbolicDifferentiator {

    private final Token[] rpnTokens;

    private static final int MAX_STACK = 1024;
    private final String[] valStack = new String[MAX_STACK];
    private final String[] derStack = new String[MAX_STACK];

    public SymbolicDifferentiator(Token[] rpnTokens) {
        this.rpnTokens = formatTokens(rpnTokens);
    }

    private static Token[] formatTokens(Token[] postfix) {
        Token[] rpn = new Token[postfix.length];
        for (int i = 0; i < postfix.length; i++) {
            rpn[i] = postfix[i].clone();
        }
        return rpn;
    }

    /**
     * Differentiates the RPN sequence symbolically.
     *
     * @param wrtVar The variable to differentiate with respect to (e.g., "x")
     * @return A 2-element array containing the function expression and its
     * derivative expression.
     */
    public String[] differentiate(String wrtVar) {
        Token[] rpn = rpnTokens;
        int sp = 0;
        for (Token t : rpn) {
            if (t.kind == Token.NUMBER) {
                valStack[sp] = formatNumber(t.value);
                derStack[sp] = "0";
                sp++;
            } else if (t.kind == Token.VARIABLE) {
                boolean isWrt = (t.name != null && t.name.equals(wrtVar));
                valStack[sp] = t.name;
                derStack[sp] = isWrt ? "1" : "0";
                sp++;
            } else if (t.kind == Token.OPERATOR) {
                sp = handleOperator(t, sp);
            } else if (t.kind == Token.METHOD || t.kind == Token.FUNCTION) {
                if (t.arity == 1) {
                    sp = handleUnaryFunction(t, sp);
                } else if (t.arity == 2) {
                    sp = handleBinaryFunction(t, sp);
                } else {
                    throw new UnsupportedOperationException("Unsupported arity for " + t.name);
                }
            }
        }
        return new String[]{valStack[0], derStack[0]};
    }

    private int handleOperator(Token t, int sp) {
        if (t.arity == 2) {
            String bVal = valStack[--sp];
            String bDer = derStack[sp];
            String aVal = valStack[--sp];
            String aDer = derStack[sp];

            switch (t.opChar) {
                case '+':
                    valStack[sp] = add(aVal, bVal);
                    derStack[sp] = add(aDer, bDer);
                    break;
                case '-':
                    valStack[sp] = sub(aVal, bVal);
                    derStack[sp] = sub(aDer, bDer);
                    break;
                case '*':
                    valStack[sp] = mul(aVal, bVal);
                    derStack[sp] = add(mul(aDer, bVal), mul(aVal, bDer));
                    break;
                case '/':
                    valStack[sp] = div(aVal, bVal);
                    derStack[sp] = div(sub(mul(aDer, bVal), mul(aVal, bDer)), pow(bVal, "2"));
                    break;
                case '^':
                    valStack[sp] = pow(aVal, bVal);
                    // Check if exponent is constant (derivative is 0) to apply simplified power rule
                    if (isZero(bDer)) {
                        derStack[sp] = mul(mul(bVal, pow(aVal, sub(bVal, "1"))), aDer);
                    } else {
                        // General case (variable exponent): a^b * (b' * ln(a) + b * a' / a)
                        String term1 = mul(bDer, "ln(" + aVal + ")");
                        String term2 = mul(bVal, div(aDer, aVal));
                        derStack[sp] = mul(valStack[sp], add(term1, term2));
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Operator not supported: " + t.opChar);
            }
            return sp + 1;
        } else if (t.arity == 1 && t.opChar == '-') {
            // Unary negation
            String val = valStack[--sp];
            String der = derStack[sp];
            valStack[sp] = neg(val);
            derStack[sp] = neg(der);
            return sp + 1;
        }
        throw new UnsupportedOperationException("Unsupported operator arity: " + t.arity);
    }

    /**
     * Differentiates the function symbolically and compiles the simplified
     * derivative string back into an RPN token array. This allows for an
     * extremely fast evaluation path without carrying over zero-terms.
     *
     * @param wrtVar The variable to differentiate with respect to (e.g., "x")
     * @return The RPN token array representing the simplified derivative.
     */
    public Token[] differentiateToRPN(String wrtVar) {
        String[] result = differentiate(wrtVar);
        String derivativeStr = result[1];

        // Feed the analytically simplified string back to the parser
        MathExpression derivativeExpr = new MathExpression(derivativeStr);
        return derivativeExpr.getCachedPostfix();
    }

    // --- Rule Registry ---
    @FunctionalInterface
    public interface UnaryRule {

        /**
         * @return An array: [functionValue, derivativeValue]
         */
        String[] apply(String uVal, String uDer);
    }

    @FunctionalInterface
    public interface BinaryRule {

        /**
         * @return An array: [functionValue, derivativeValue]
         */
        String[] apply(String uVal, String uDer, String vVal, String vDer);
    }

    private static final Map<String, UnaryRule> UNARY_RULES = new HashMap<>();
    private static final Map<String, BinaryRule> BINARY_RULES = new HashMap<>();

    /**
     * Expose this method to allow users to register custom function
     * derivatives.
     */
    public static void registerUnaryRule(String functionName, UnaryRule rule) {
        UNARY_RULES.put(functionName, rule);
    }

    /**
     * Expose this method to allow users to register custom binary function
     * derivatives.
     */
    public static void registerBinaryRule(String functionName, BinaryRule rule) {
        BINARY_RULES.put(functionName, rule);
    }

    static {
        // --- Unary Rules ---

        addUnary((u, du) -> new String[]{"sin(" + u + ")", mul("cos(" + u + ")", du)}, Declarations.SIN);
        addUnary((u, du) -> new String[]{"cos(" + u + ")", mul(neg("sin(" + u + ")"), du)}, Declarations.COS);
        addUnary((u, du) -> new String[]{"tan(" + u + ")", div(du, pow("cos(" + u + ")", "2"))}, Declarations.TAN);

        addUnary((u, du) -> new String[]{"asin(" + u + ")", div(du, "sqrt(" + sub("1", pow(u, "2")) + ")")},
                Declarations.ARC_SIN, Declarations.ARC_SIN_ALT);
        addUnary((u, du) -> new String[]{"acos(" + u + ")", div(neg(du), "sqrt(" + sub("1", pow(u, "2")) + ")")},
                Declarations.ARC_COS, Declarations.ARC_COS_ALT);
        addUnary((u, du) -> new String[]{"atan(" + u + ")", div(du, add("1", pow(u, "2")))},
                Declarations.ARC_TAN, Declarations.ARC_TAN_ALT);

        addUnary((u, du) -> new String[]{"sec(" + u + ")", mul(du, mul("sec(" + u + ")", "tan(" + u + ")"))}, Declarations.SEC);
        addUnary((u, du) -> new String[]{"csc(" + u + ")", mul(neg(du), mul("csc(" + u + ")", "cot(" + u + ")"))},
                "cosec", Declarations.COSEC);
        addUnary((u, du) -> new String[]{"cot(" + u + ")", div(neg(du), pow("sin(" + u + ")", "2"))}, Declarations.COT);

        addUnary((u, du) -> new String[]{"asec(" + u + ")", div(du, mul(abs(u), "sqrt(" + sub(pow(u, "2"), "1") + ")"))},
                Declarations.ARC_SEC, Declarations.ARC_SEC_ALT);
        addUnary((u, du) -> new String[]{"acsc(" + u + ")", div(neg(du), mul(abs(u), "sqrt(" + sub(pow(u, "2"), "1") + ")"))},
                Declarations.ARC_COSEC, Declarations.ARC_COSEC_ALT);
        addUnary((u, du) -> new String[]{"acot(" + u + ")", div(neg(du), add("1", pow(u, "2")))},
                Declarations.ARC_COT, Declarations.ARC_COT_ALT);

        addUnary((u, du) -> new String[]{"sinh(" + u + ")", mul("cosh(" + u + ")", du)}, Declarations.SINH);
        addUnary((u, du) -> new String[]{"cosh(" + u + ")", mul("sinh(" + u + ")", du)}, Declarations.COSH);
        addUnary((u, du) -> new String[]{"tanh(" + u + ")", mul(div("1", pow("cosh(" + u + ")", "2")), du)}, Declarations.TANH);

        addUnary((u, du) -> new String[]{"asinh(" + u + ")", div(du, "sqrt(" + add(pow(u, "2"), "1") + ")")},
                Declarations.ARC_SINH, Declarations.ARC_SINH_ALT);
        addUnary((u, du) -> new String[]{"acosh(" + u + ")", div(du, "sqrt(" + sub(pow(u, "2"), "1") + ")")},
                Declarations.ARC_COSH, Declarations.ARC_COSH_ALT);
        addUnary((u, du) -> new String[]{"atanh(" + u + ")", div(du, sub("1", pow(u, "2")))},
                Declarations.ARC_TANH, Declarations.ARC_TANH_ALT);

        addUnary((u, du) -> new String[]{"asech(" + u + ")", div(neg(du), mul(u, "sqrt(" + sub("1", pow(u, "2")) + ")"))},
                Declarations.ARC_SECH, Declarations.ARC_SECH_ALT);
        addUnary((u, du) -> new String[]{"acsch(" + u + ")", div(neg(du), mul(abs(u), "sqrt(" + add(pow(u, "2"), "1") + ")"))},
                Declarations.ARC_COSECH, Declarations.ARC_COSECH_ALT);
        addUnary((u, du) -> new String[]{"acoth(" + u + ")", div(neg(du), sub(pow(u, "2"), "1"))},
                Declarations.ARC_COTH, Declarations.ARC_COTH_ALT);

        addUnary((u, du) -> new String[]{"sqrt(" + u + ")", div(du, mul("2", "sqrt(" + u + ")"))}, Declarations.SQRT);
        addUnary((u, du) -> new String[]{"cbrt(" + u + ")", div(du, mul("3", pow("cbrt(" + u + ")", "2")))}, Declarations.CBRT);
        addUnary((u, du) -> new String[]{"exp(" + u + ")", mul("exp(" + u + ")", du)}, Declarations.EXP);
        addUnary((u, du) -> new String[]{"ln(" + u + ")", div(du, u)}, Declarations.LN);
        addUnary((u, du) -> new String[]{"lg(" + u + ")", div(du, mul(u, "ln(10)"))}, "log", Declarations.LG);
        addUnary((u, du) -> new String[]{"abs(" + u + ")", mul(div(u, "abs(" + u + ")"), du)}, "abs");

        // --- Binary Rules ---
        BINARY_RULES.put(Declarations.POW, (u, du, v, dv) -> {
            String val = "pow(" + u + ", " + v + ")";
            String der;
            if (isZero(dv)) {
                der = mul(mul(v, "pow(" + u + ", " + sub(v, "1") + ")"), du);
            } else {
                String term1 = mul(dv, "ln(" + u + ")");
                String term2 = mul(v, div(du, u));
                der = mul(val, add(term1, term2));
            }
            return new String[]{val, der};
        });

        BINARY_RULES.put(Declarations.LOG, (u, du, v, dv) -> {
            String lnU = "ln(" + u + ")";
            String lnV = "ln(" + v + ")";
            String val = div(lnU, lnV);
            String num = sub(mul(div(du, u), lnV), mul(lnU, div(dv, v)));
            String der = div(num, pow(lnV, "2"));
            return new String[]{val, der};
        });

        BINARY_RULES.put("atan2", (u, du, v, dv) -> {
            String val = "atan2(" + u + ", " + v + ")";
            String denom = add(pow(u, "2"), pow(v, "2"));
            String der = div(sub(mul(du, v), mul(u, dv)), denom);
            return new String[]{val, der};
        });

        BINARY_RULES.put("hypot", (u, du, v, dv) -> {
            String val = "hypot(" + u + ", " + v + ")";
            String der = div(add(mul(u, du), mul(v, dv)), val);
            return new String[]{val, der};
        });
    }

    private static void addUnary(UnaryRule rule, String... names) {
        for (String name : names) {
            UNARY_RULES.put(name, rule);
        }
    }

    // ... (Constructor, differentiate, and differentiateToRPN remain exactly the same) ...
    private int handleUnaryFunction(Token t, int sp) {
        UnaryRule rule = UNARY_RULES.get(t.name);
        if (rule == null) {
            throw new UnsupportedOperationException("Symbolic AD not implemented for: " + t.name);
        }

        String uVal = valStack[--sp];
        String uDer = derStack[sp];

        String[] result = rule.apply(uVal, uDer);
        valStack[sp] = result[0];
        derStack[sp] = result[1];

        return sp + 1;
    }

    private int handleBinaryFunction(Token t, int sp) {
        BinaryRule rule = BINARY_RULES.get(t.name);
        if (rule == null) {
            throw new UnsupportedOperationException("2-arg function not supported: " + t.name);
        }

        String vVal = valStack[--sp];
        String vDer = derStack[sp];
        String uVal = valStack[--sp];
        String uDer = derStack[sp];

        String[] result = rule.apply(uVal, uDer, vVal, vDer);
        valStack[sp] = result[0];
        derStack[sp] = result[1];

        return sp + 1;
    }

    // --- Algebraic Simplification Helpers (Now static) ---
    private static boolean isZero(String s) {
        return s.equals("0") || s.equals("0.0");
    }

    private static boolean isOne(String s) {
        return s.equals("1") || s.equals("1.0");
    }

    // Note: If you have locale issues, enforce Locale.US here: String.format(Locale.US, "%f", d)
    private static String formatNumber(double d) {
        if (d == (long) d) {
            return String.valueOf((long) d);
        }
        return String.valueOf(d);
    }

    private static String add(String a, String b) {
        if (isZero(a)) {
            return b;
        }
        if (isZero(b)) {
            return a;
        }
        if (a.equals(b)) {
            return mul("2", a);
        }
        return "(" + a + " + " + b + ")";
    }

    private static String sub(String a, String b) {
        if (isZero(b)) {
            return a;
        }
        if (isZero(a)) {
            return neg(b);
        }
        if (a.equals(b)) {
            return "0";
        }
        return "(" + a + " - " + b + ")";
    }

    private static String mul(String a, String b) {
        if (isZero(a) || isZero(b)) {
            return "0";
        }
        if (isOne(a)) {
            return b;
        }
        if (isOne(b)) {
            return a;
        }
        if (a.equals("-1") || a.equals("-1.0")) {
            return neg(b);
        }
        if (b.equals("-1") || b.equals("-1.0")) {
            return neg(a);
        }
        return "(" + a + " * " + b + ")";
    }

    private static String div(String a, String b) {
        if (isZero(a)) {
            return "0";
        }
        if (isOne(b)) {
            return a;
        }
        if (a.equals(b)) {
            return "1";
        }
        return "(" + a + " / " + b + ")";
    }

    private static String pow(String a, String b) {
        if (isZero(b)) {
            return "1";
        }
        if (isOne(b)) {
            return a;
        }
        if (isZero(a)) {
            return "0";
        }
        return "(" + a + "^" + b + ")";
    }

    private static String neg(String a) {
        if (isZero(a)) {
            return "0";
        }
        if (a.startsWith("-")) {
            return a.substring(1);
        }
        if (a.matches("^[a-zA-Z0-9_.]+$") || a.matches("^[a-zA-Z]+\\(.*\\)$")) {
            return "-" + a;
        }
        return "-(" + a + ")";
    }

    private static String abs(String a) {
        if (isZero(a)) {
            return "0";
        }
        return "abs(" + a + ")";
    }
    
       public static void main(String[] args) {
        String expr = "x^3+3*x^2-5*x-8*atan2(2*x, 3)";
        MathExpression me = new MathExpression(expr);
        SymbolicDifferentiator sd = new SymbolicDifferentiator(me.getCachedPostfix());

        String wrtVar = "x";
        String[] stringResult = sd.differentiate(wrtVar);

        System.out.println("Function: " + expr);
        System.out.println("Symbolic Evaluation f(" + wrtVar + "): " + stringResult[0]);
        System.out.println("Symbolic Derivative f'(" + wrtVar + "): " + stringResult[1]);

        // Get compiled RPN Tokens
        Token[] rpnTokens = sd.differentiateToRPN(wrtVar);
        System.out.println("\nCompiled Derivative RPN Tokens Length: " + rpnTokens.length);
        
        // (Optional check) You can verify evaluation using the RPN evaluator or MathExpression
        // Double evaluation using MathExpression
        MathExpression derivativeTester = new MathExpression(stringResult[1]);
        derivativeTester.setValue("x", 2.0);
        System.out.println("f'(2) calculated via compiled derivative RPN: " + derivativeTester.solve());
    }
}
