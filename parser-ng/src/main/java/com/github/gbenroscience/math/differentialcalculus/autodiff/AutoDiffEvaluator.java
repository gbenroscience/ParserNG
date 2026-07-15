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
package com.github.gbenroscience.math.differentialcalculus.autodiff;

/**
 *
 * @author oluwagbemirojiboye
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;

/**
 * GC-Free Forward Mode Automatic Differentiation Evaluator for Declarations.
 * Now supports a much wider range of Declarations built-in functions.
 *
 * <p><b>Thread-safety:</b> this class is intentionally <i>not</i> thread-safe.
 * {@code valStack}/{@code derStack} are pre-sized once and reused across every
 * call to {@link #evaluateRPN} to guarantee zero per-call heap allocation on
 * the hot path -- that reuse is what makes it fast, but it also means two
 * threads calling {@code evaluateRPN} concurrently on the <b>same instance</b>
 * will corrupt each other's stack. Give each thread (or each pooled worker)
 * its own {@code AutoDiffEvaluator} instance; do not share one across
 * threads. The instance itself is safe to reuse sequentially, e.g. in a tight
 * single-threaded loop evaluating many points -- that is the intended usage.
 */
public class AutoDiffEvaluator {

    private final Token[] rpnTokens;

    // Sized once, in the constructor, to the smallest bound that can never
    // overflow: RPN evaluation can never need more stack slots than there are
    // tokens. This replaces a previous fixed MAX_STACK=1024 array, which both
    // wasted ~16KB/instance for small expressions and threw an unguarded,
    // unhelpful ArrayIndexOutOfBoundsException for anything larger. Sizing
    // once here preserves the zero-allocation-per-call design.
    private final double[] valStack;
    private final double[] derStack;

    public AutoDiffEvaluator(MathExpression me) { 
        if (me == null || me.getCachedPostfix() == null || me.getCachedPostfix().length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        this.rpnTokens = formatTokens(me.getCachedPostfix());
        this.valStack = new double[this.rpnTokens.length + 1];
        this.derStack = new double[this.rpnTokens.length + 1];
    }

   private static Token[] formatTokens(Token[] postfix) {
        Token[] rpn = new Token[postfix.length];
        for (int i = 0; i < postfix.length; i++) {
           Token t = postfix[i].clone(); 
           t.name = (t.name != null && (t.name.endsWith("_rad") || t.name.endsWith("_grad") || t.name.endsWith("_deg"))) ? t.name.substring(0, t.name.lastIndexOf("_")) : t.name;
            rpn[i] = t;
        }
        return rpn;
    }

    public void evaluateRPN(String wrtVar, double evalPoint, double resultOut[]) {
        Token[] rpn = rpnTokens;
        int sp = 0;
        for (Token t : rpn) {
            if (t.kind == Token.NUMBER) {
                valStack[sp] = t.value;
                derStack[sp] = 0.0;
                sp++;
            } else if (t.kind == Token.VARIABLE) {
                boolean isWrt = (t.name != null && t.name.equals(wrtVar));
                valStack[sp] = isWrt ? evalPoint : (t.v != null ? t.v.getValue() : t.value);
                derStack[sp] = isWrt ? 1.0 : 0.0;
                sp++;
            } else if (t.kind == Token.OPERATOR) {
                sp = handleOperator(t, sp); // Reassign updated stack pointer
            } else if (t.kind == Token.METHOD || t.kind == Token.FUNCTION) {
                if (t.arity == 1) {
                    sp = handleUnaryFunction(t, sp); // Reassign updated stack pointer
                } else if (t.arity == 2) {
                    sp = handleBinaryFunction(t, sp); // Reassign updated stack pointer
                } else {
                    throw new UnsupportedOperationException("Unsupported arity for " + t.name);
                }
            } else {
                throw new UnsupportedOperationException("Unrecognized token kind: " + t.kind
                        + (t.name != null ? " (" + t.name + ")" : ""));
            }
        }

        if (sp != 1) {
            throw new IllegalStateException(
                    "Malformed RPN expression: expected exactly one result on the stack, found " + sp);
        }

        resultOut[0] = valStack[0];
        resultOut[1] = derStack[0];
    }

    public double[] evaluateRPN(String wrtVar, double evalPoint) {
        double[] resultOut = new double[2];
        evaluateRPN(wrtVar, evalPoint, resultOut);
        return resultOut;
    }

    private static void requireOperands(int sp, int needed, String opName) {
        if (sp < needed) {
            throw new IllegalStateException(
                    "Malformed RPN expression: '" + opName + "' requires " + needed
                            + " operand(s) but only " + sp + " available on the stack");
        }
    }

    /**
     * Shared power-rule derivative, used identically by the binary '^'
     * operator and the 2-arg pow(u,v) function. Previously duplicated
     * verbatim in both call sites -- a maintenance hazard where a future fix
     * to one copy could easily be forgotten in the other.
     */
    private static double powDerivative(double uVal, double uDer, double vVal, double vDer, double powResult) {
        if (vDer == 0.0) {
            // Constant exponent
            return vVal * Math.pow(uVal, vVal - 1.0) * uDer;
        }
        // Variable exponent
        if (uVal <= 0.0 && vVal != Math.floor(vVal)) {
            throw new ArithmeticException("Non-real pow");
        }
        return powResult * (vDer * Math.log(uVal) + vVal * uDer / uVal);
    }

    private int handleOperator(Token t, int sp) {
        if (t.arity == 2) {
            requireOperands(sp, 2, String.valueOf(t.opChar));
            double bVal = valStack[--sp];
            double bDer = derStack[sp];
            double aVal = valStack[--sp];
            double aDer = derStack[sp];

            switch (t.opChar) {
                case '+':
                    valStack[sp] = aVal + bVal;
                    derStack[sp] = aDer + bDer;
                    break;
                case '-':
                    valStack[sp] = aVal - bVal;
                    derStack[sp] = aDer - bDer;
                    break;
                case '*':
                    valStack[sp] = aVal * bVal;
                    derStack[sp] = aDer * bVal + aVal * bDer;
                    break;
                case '/':
                    if (bVal == 0.0) {
                        throw new ArithmeticException("Division by zero");
                    }
                    valStack[sp] = aVal / bVal;
                    derStack[sp] = (aDer * bVal - aVal * bDer) / (bVal * bVal);
                    break;
                case '^':
                    valStack[sp] = Math.pow(aVal, bVal);
                    derStack[sp] = powDerivative(aVal, aDer, bVal, bDer, valStack[sp]);
                    break;
                default:
                    throw new UnsupportedOperationException("Operator not supported: " + t.opChar);
            }
            return sp + 1;
        } else if (t.arity == 1) {
            // Unary minus
            if (t.opChar == '-') {
                requireOperands(sp, 1, "unary -");
                double val = valStack[--sp];
                double der = derStack[sp];
                valStack[sp] = -val;
                derStack[sp] = -der;
                return sp + 1;
            } else {
                throw new UnsupportedOperationException("Unary operator not supported: " + t.opChar);
            }
        }
        return sp;
    }

    private int handleUnaryFunction(Token t, int sp) {
        requireOperands(sp, 1, t.name);
        double argVal = valStack[--sp];
        double argDer = derStack[sp];
        double val, der;

        switch (t.name) {
            case Declarations.SIN:
                val = Math.sin(argVal);
                der = Math.cos(argVal) * argDer;
                break;
            case Declarations.COS:
                val = Math.cos(argVal);
                der = -Math.sin(argVal) * argDer;
                break;
            case Declarations.TAN:
                val = Math.tan(argVal);
                double cos = Math.cos(argVal);
                der = argDer / (cos * cos);
                break;

            // Inverse trig
            case Declarations.ARC_SIN:
            case Declarations.ARC_SIN_ALT:
                if (Math.abs(argVal) > 1.0) {
                    throw new ArithmeticException("asin domain");
                }
                val = Math.asin(argVal);
                der = argDer / Math.sqrt(1 - argVal * argVal);
                break;
            case Declarations.ARC_COS:
            case Declarations.ARC_COS_ALT:
                if (Math.abs(argVal) > 1.0) {
                    throw new ArithmeticException("acos domain");
                }
                val = Math.acos(argVal);
                der = -argDer / Math.sqrt(1 - argVal * argVal);
                break;
            case Declarations.ARC_TAN:
            case Declarations.ARC_TAN_ALT:
                val = Math.atan(argVal);
                der = argDer / (1 + argVal * argVal);
                break;

            // Reciprocal trig
            case Declarations.SEC:
                double c = Math.cos(argVal);
                if (Math.abs(c) < 1e-15) {
                    throw new ArithmeticException("sec undefined");
                }
                val = 1.0 / c;
                der = argDer * (Math.tan(argVal) / c);   // sec * tan
                break;
            case Declarations.COSEC: // csc
                double s = Math.sin(argVal);
                if (Math.abs(s) < 1e-15) {
                    throw new ArithmeticException("csc undefined");
                }
                val = 1.0 / s;
                der = -argDer * (Math.cos(argVal) / (s * s));
                break;
            case Declarations.COT:
                s = Math.sin(argVal);
                if (Math.abs(s) < 1e-15) {
                    throw new ArithmeticException("cot undefined");
                }
                val = Math.cos(argVal) / s;
                der = -argDer / (s * s);
                break;

            // Inverse reciprocal trig
            case Declarations.ARC_SEC:
            case Declarations.ARC_SEC_ALT:
                if (Math.abs(argVal) < 1.0) {
                    throw new ArithmeticException("asec domain");
                }
                val = Math.acos(1.0 / argVal);
                der = argDer / (Math.abs(argVal) * Math.sqrt(argVal * argVal - 1));
                break;

            case Declarations.ARC_COSEC:
            case Declarations.ARC_COSEC_ALT:
                if (Math.abs(argVal) < 1.0) {
                    throw new ArithmeticException("acsc domain");
                }
                val = Math.asin(1.0 / argVal);
                der = -argDer / (Math.abs(argVal) * Math.sqrt(argVal * argVal - 1));
                break;

            case Declarations.ARC_COT:
            case Declarations.ARC_COT_ALT:
                val = Math.atan(1.0 / argVal);   // or π/2 - atan(argVal) adjustment if needed
                der = -argDer / (1 + argVal * argVal);
                break;

            // Hyperbolic
            case Declarations.SINH:
                val = Math.sinh(argVal);
                der = Math.cosh(argVal) * argDer;
                break;
            case Declarations.COSH:
                val = Math.cosh(argVal);
                der = Math.sinh(argVal) * argDer;
                break;
            case Declarations.TANH:
                val = Math.tanh(argVal);
                double sech2 = 1.0 / (Math.cosh(argVal) * Math.cosh(argVal));
                der = sech2 * argDer;
                break;

            // Inverse hyperbolic
            case Declarations.ARC_SINH:
            case Declarations.ARC_SINH_ALT:
                val = Math.log(argVal + Math.sqrt(argVal * argVal + 1));
                der = argDer / Math.sqrt(argVal * argVal + 1);
                break;

            case Declarations.ARC_COSH:
            case Declarations.ARC_COSH_ALT:
                if (argVal < 1.0) {
                    throw new ArithmeticException("acosh domain");
                }
                val = Math.log(argVal + Math.sqrt(argVal * argVal - 1));
                der = argDer / Math.sqrt(argVal * argVal - 1);
                break;

            case Declarations.ARC_TANH:
            case Declarations.ARC_TANH_ALT:
                if (Math.abs(argVal) >= 1.0) {
                    throw new ArithmeticException("atanh domain");
                }
                val = 0.5 * Math.log((1 + argVal) / (1 - argVal));
                der = argDer / (1 - argVal * argVal);
                break;

            case Declarations.ARC_SECH:
            case Declarations.ARC_SECH_ALT:
                if (argVal <= 0 || argVal > 1.0) {
                    throw new ArithmeticException("asech domain");
                }
                val = Math.log(1.0 / argVal + Math.sqrt(1.0 / (argVal * argVal) - 1));
                der = -argDer / (argVal * Math.sqrt(1 - argVal * argVal));
                break;

            case Declarations.ARC_COSECH:
            case Declarations.ARC_COSECH_ALT:
                if (argVal == 0.0) {
                    throw new ArithmeticException("acsch domain");
                }
                val = Math.log(1.0 / argVal + Math.sqrt(1.0 / (argVal * argVal) + 1));
                der = -argDer / (Math.abs(argVal) * Math.sqrt(argVal * argVal + 1));
                break;

            case Declarations.ARC_COTH:
            case Declarations.ARC_COTH_ALT:
                if (Math.abs(argVal) <= 1.0) {
                    throw new ArithmeticException("acoth domain");
                }
                val = 0.5 * Math.log((argVal + 1) / (argVal - 1));
                der = -argDer / (argVal * argVal - 1);
                break;

            // Roots & Exponential / Log
            case Declarations.SQRT:
                if (argVal < 0) {
                    throw new ArithmeticException("sqrt domain");
                }
                val = Math.sqrt(argVal);
                der = argDer / (2 * val);
                break;
            case Declarations.CBRT:
                val = Math.cbrt(argVal);
                der = argDer / (3 * Math.pow(val, 2));
                break;
            case Declarations.EXP:
                val = Math.exp(argVal);
                der = val * argDer;
                break;
            case Declarations.LN:
                if (argVal <= 0) {
                    throw new ArithmeticException("ln domain");
                }
                val = Math.log(argVal);
                der = argDer / argVal;
                break;
            case Declarations.LG:
                if (argVal <= 0) {
                    throw new ArithmeticException("log domain");
                }
                val = Math.log10(argVal);
                der = argDer / (argVal * Math.log(10));
                break;

            case "abs":
                val = Math.abs(argVal);
                der = (argVal > 0.0) ? argDer : (argVal < 0.0) ? -argDer : 0.0;
                break;

            default:
                throw new UnsupportedOperationException("AD not implemented for: " + t.name);
        }

        valStack[sp] = val;
        derStack[sp] = der;
        return sp + 1;
    }

    private int handleBinaryFunction(Token t, int sp) {
        requireOperands(sp, 2, t.name);
        double vVal = valStack[--sp];
        double vDer = derStack[sp];
        double uVal = valStack[--sp];
        double uDer = derStack[sp];

        switch (t.name) {
            case Declarations.POW:
                valStack[sp] = Math.pow(uVal, vVal);
                derStack[sp] = powDerivative(uVal, uDer, vVal, vDer, valStack[sp]);
                break;

            case Declarations.LOG:
                // Represents log_v(u). Value is log(u)/log(v)
                if (uVal <= 0 || vVal <= 0 || vVal == 1.0) {
                    throw new ArithmeticException("log domain/base exception");
                }
                double lnU = Math.log(uVal);
                double lnV = Math.log(vVal);

                valStack[sp] = lnU / lnV;
                derStack[sp] = ((uDer / uVal) * lnV - lnU * (vDer / vVal)) / (lnV * lnV);
                break;

            case "atan2":
                valStack[sp] = Math.atan2(uVal, vVal);
                double denom = (uVal * uVal) + (vVal * vVal);
                derStack[sp] = (uDer * vVal - uVal * vDer) / denom;
                break;

            case "hypot":
                valStack[sp] = Math.hypot(uVal, vVal);
                derStack[sp] = (uVal * uDer + vVal * vDer) / valStack[sp];
                break;

            default:
                throw new UnsupportedOperationException("2-arg function not supported: " + t.name);
        }

        return sp + 1;
    }

    public static void main(String[] args) {
        String expr = "x^3+3*x^2-5*x-8*atan2(2*x, 3)";
        MathExpression me = new MathExpression(expr);
        AutoDiffEvaluator ad = new AutoDiffEvaluator(me);

        // Evaluate at x = 2.0
        double x = 2.0;
        double[] out = new double[2];
        ad.evaluateRPN("x", x, out);

        System.out.println("Function: "+expr);
        System.out.println("Evaluating at x = " + x);

        System.out.println("Value f(x): " + out[0]);

        System.out.println("Derivative f'(x): " + out[1]);

        MathExpression mee = new MathExpression("autodiff(@(x)x^3+3*x^2-5*x-8*atan2(2*x, 3),2)");
        System.out.println(mee.solveGeneric());
    }
}