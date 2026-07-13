/*
 * Copyright 2026 oluwagbemirojiboye.
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
package com.github.gbenroscience.math.differentialcalculus.symbolic;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;

/**
 * GC-Free Forward Mode Symbolic Differentiation Evaluator for Declarations.
 * Generates string representations of derivatives analytically, with the 
 * ability to compile them back into RPN Tokens for fast repeated evaluation.
 * 
 * @author oluwagbemirojiboye
 */
public class SymbolicDifferentiator1 {

    private final Token[] rpnTokens;

    private static final int MAX_STACK = 1024;
    private final String[] valStack = new String[MAX_STACK];
    private final String[] derStack = new String[MAX_STACK];

    public SymbolicDifferentiator1(Token[] rpnTokens) {
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
     * @return A 2-element array containing the function expression and its derivative expression.
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
                    if (isZero(bDer)) {
                        // Constant exponent rule: b * a^(b-1) * a'
                        derStack[sp] = mul(mul(bVal, pow(aVal, sub(bVal, "1"))), aDer);
                    } else {
                        // Variable exponent rule: a^b * (b' * ln(a) + b * a' / a)
                        String term1 = mul(bDer, "ln(" + aVal + ")");
                        String term2 = mul(bVal, div(aDer, aVal));
                        derStack[sp] = mul(valStack[sp], add(term1, term2));
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Operator not supported: " + t.opChar);
            }
            return sp + 1;
        } else if (t.arity == 1) {
            if (t.opChar == '-') {
                String val = valStack[--sp];
                String der = derStack[sp];
                valStack[sp] = neg(val);
                derStack[sp] = neg(der);
                return sp + 1;
            } else {
                throw new UnsupportedOperationException("Unary operator not supported: " + t.opChar);
            }
        }
        return sp;
    }

    private int handleUnaryFunction(Token t, int sp) {
        String uVal = valStack[--sp];
        String uDer = derStack[sp];
        String val, der;

        switch (t.name) {
            case Declarations.SIN:
                val = "sin(" + uVal + ")";
                der = mul("cos(" + uVal + ")", uDer);
                break;
            case Declarations.COS:
                val = "cos(" + uVal + ")";
                der = mul(neg("sin(" + uVal + ")"), uDer);
                break;
            case Declarations.TAN:
                val = "tan(" + uVal + ")";
                der = div(uDer, pow("cos(" + uVal + ")", "2"));
                break;
            case Declarations.ARC_SIN:
            case Declarations.ARC_SIN_ALT:
                val = "asin(" + uVal + ")";
                der = div(uDer, "sqrt(" + sub("1", pow(uVal, "2")) + ")");
                break;
            case Declarations.ARC_COS:
            case Declarations.ARC_COS_ALT:
                val = "acos(" + uVal + ")";
                der = div(neg(uDer), "sqrt(" + sub("1", pow(uVal, "2")) + ")");
                break;
            case Declarations.ARC_TAN:
            case Declarations.ARC_TAN_ALT:
                val = "atan(" + uVal + ")";
                der = div(uDer, add("1", pow(uVal, "2")));
                break;
            case Declarations.SEC:
                val = "sec(" + uVal + ")";
                der = mul(uDer, mul("sec(" + uVal + ")", "tan(" + uVal + ")"));
                break;
            case "cosec":
            case Declarations.COSEC:
                val = "csc(" + uVal + ")";
                der = mul(neg(uDer), mul("csc(" + uVal + ")", "cot(" + uVal + ")"));
                break;
            case Declarations.COT:
                val = "cot(" + uVal + ")";
                der = div(neg(uDer), pow("sin(" + uVal + ")", "2"));
                break;
            case Declarations.ARC_SEC:
            case Declarations.ARC_SEC_ALT:
                val = "asec(" + uVal + ")";
                der = div(uDer, mul(abs(uVal), "sqrt(" + sub(pow(uVal, "2"), "1") + ")"));
                break;
            case Declarations.ARC_COSEC:
            case Declarations.ARC_COSEC_ALT:
                val = "acsc(" + uVal + ")";
                der = div(neg(uDer), mul(abs(uVal), "sqrt(" + sub(pow(uVal, "2"), "1") + ")"));
                break;
            case Declarations.ARC_COT:
            case Declarations.ARC_COT_ALT:
                val = "acot(" + uVal + ")";
                der = div(neg(uDer), add("1", pow(uVal, "2")));
                break;
            case Declarations.SINH:
                val = "sinh(" + uVal + ")";
                der = mul("cosh(" + uVal + ")", uDer);
                break;
            case Declarations.COSH:
                val = "cosh(" + uVal + ")";
                der = mul("sinh(" + uVal + ")", uDer);
                break;
            case Declarations.TANH:
                val = "tanh(" + uVal + ")";
                der = mul(div("1", pow("cosh(" + uVal + ")", "2")), uDer);
                break;
            case Declarations.ARC_SINH:
            case Declarations.ARC_SINH_ALT:
                val = "asinh(" + uVal + ")";
                der = div(uDer, "sqrt(" + add(pow(uVal, "2"), "1") + ")");
                break;
            case Declarations.ARC_COSH:
            case Declarations.ARC_COSH_ALT:
                val = "acosh(" + uVal + ")";
                der = div(uDer, "sqrt(" + sub(pow(uVal, "2"), "1") + ")");
                break;
            case Declarations.ARC_TANH:
            case Declarations.ARC_TANH_ALT:
                val = "atanh(" + uVal + ")";
                der = div(uDer, sub("1", pow(uVal, "2")));
                break;
            case Declarations.ARC_SECH:
            case Declarations.ARC_SECH_ALT:
                val = "asech(" + uVal + ")";
                der = div(neg(uDer), mul(uVal, "sqrt(" + sub("1", pow(uVal, "2")) + ")"));
                break;
            case Declarations.ARC_COSECH:
            case Declarations.ARC_COSECH_ALT:
                val = "acsch(" + uVal + ")";
                der = div(neg(uDer), mul(abs(uVal), "sqrt(" + add(pow(uVal, "2"), "1") + ")"));
                break;
            case Declarations.ARC_COTH:
            case Declarations.ARC_COTH_ALT:
                val = "acoth(" + uVal + ")";
                der = div(neg(uDer), sub(pow(uVal, "2"), "1"));
                break;
            case Declarations.SQRT:
                val = "sqrt(" + uVal + ")";
                der = div(uDer, mul("2", val));
                break;
            case Declarations.CBRT:
                val = "cbrt(" + uVal + ")";
                der = div(uDer, mul("3", pow(val, "2")));
                break;
            case Declarations.EXP:
                val = "exp(" + uVal + ")";
                der = mul(val, uDer);
                break;
            case Declarations.LN:
                val = "ln(" + uVal + ")";
                der = div(uDer, uVal);
                break;
            case "log":
            case Declarations.LG:
                val = "lg(" + uVal + ")";
                der = div(uDer, mul(uVal, "ln(10)"));
                break;
            case "abs":
                val = "abs(" + uVal + ")";
                der = mul(div(uVal, val), uDer);
                break;
            default:
                throw new UnsupportedOperationException("Symbolic AD not implemented for: " + t.name);
        }

        valStack[sp] = val;
        derStack[sp] = der;
        return sp + 1;
    }

    private int handleBinaryFunction(Token t, int sp) {
        String vVal = valStack[--sp];
        String vDer = derStack[sp];
        String uVal = valStack[--sp];
        String uDer = derStack[sp];

        switch (t.name) {
            case Declarations.POW:
                valStack[sp] = "pow(" + uVal + ", " + vVal + ")";
                if (isZero(vDer)) {
                    derStack[sp] = mul(mul(vVal, "pow(" + uVal + ", " + sub(vVal, "1") + ")"), uDer);
                } else {
                    String term1 = mul(vDer, "ln(" + uVal + ")");
                    String term2 = mul(vVal, div(uDer, uVal));
                    derStack[sp] = mul(valStack[sp], add(term1, term2));
                }
                break;
            case Declarations.LOG:
                // Parseable log form: ln(u)/ln(v)
                valStack[sp] = div("ln(" + uVal + ")", "ln(" + vVal + ")");
                String lnU = "ln(" + uVal + ")";
                String lnV = "ln(" + vVal + ")";
                String num = sub(mul(div(uDer, uVal), lnV), mul(lnU, div(vDer, vVal)));
                derStack[sp] = div(num, pow(lnV, "2"));
                break;
            case "atan2":
                valStack[sp] = "atan2(" + uVal + ", " + vVal + ")";
                String denom = add(pow(uVal, "2"), pow(vVal, "2"));
                derStack[sp] = div(sub(mul(uDer, vVal), mul(uVal, vDer)), denom);
                break;
            case "hypot":
                valStack[sp] = "hypot(" + uVal + ", " + vVal + ")";
                derStack[sp] = div(add(mul(uVal, uDer), mul(vVal, vDer)), valStack[sp]);
                break;
            default:
                throw new UnsupportedOperationException("2-arg function not supported: " + t.name);
        }

        return sp + 1;
    }

    // --- Algebraic Simplification Helpers ---

    private boolean isZero(String s) {
        return s.equals("0") || s.equals("0.0");
    }

    private boolean isOne(String s) {
        return s.equals("1") || s.equals("1.0");
    }

    private String formatNumber(double d) {
        if (d == (long) d) {
            return String.valueOf((long) d);
        }
        return String.valueOf(d);
    }

    private String add(String a, String b) {
        if (isZero(a)) return b;
        if (isZero(b)) return a;
        if (a.equals(b)) return mul("2", a);
        return "(" + a + " + " + b + ")";
    }

    private String sub(String a, String b) {
        if (isZero(b)) return a;
        if (isZero(a)) return neg(b);
        if (a.equals(b)) return "0";
        return "(" + a + " - " + b + ")";
    }

    private String mul(String a, String b) {
        if (isZero(a) || isZero(b)) return "0";
        if (isOne(a)) return b;
        if (isOne(b)) return a;
        if (a.equals("-1") || a.equals("-1.0")) return neg(b);
        if (b.equals("-1") || b.equals("-1.0")) return neg(a);
        return "(" + a + " * " + b + ")";
    }

    private String div(String a, String b) {
        if (isZero(a)) return "0";
        if (isOne(b)) return a;
        if (a.equals(b)) return "1";
        return "(" + a + " / " + b + ")";
    }

    private String pow(String a, String b) {
        if (isZero(b)) return "1";
        if (isOne(b)) return a;
        if (isZero(a)) return "0";
        return "(" + a + "^" + b + ")";
    }

    private String neg(String a) {
        if (isZero(a)) return "0";
        if (a.startsWith("-")) return a.substring(1); // Double negation
        // Wrap composite expressions in parens
        if (a.matches("^[a-zA-Z0-9_.]+$") || a.matches("^[a-zA-Z]+\\(.*\\)$")) return "-" + a;
        return "-(" + a + ")";
    }

    private String abs(String a) {
        if (isZero(a)) return "0";
        return "abs(" + a + ")";
    }

    // --- Main Method Example ---
    
    public static void main(String[] args) {
        String expr = "x^3+3*x^2-5*x-8*atan2(2*x, 3)";
        MathExpression me = new MathExpression(expr);
        SymbolicDifferentiator1 sd = new SymbolicDifferentiator1(me.getCachedPostfix());

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