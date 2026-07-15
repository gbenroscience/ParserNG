package com.github.gbenroscience.math.differentialcalculus.symbolic;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Symbolic Differentiator with improved expression simplification and
 * parenthesis control.
 *
 * Thread-safety: instances are immutable after construction (the RPN token
 * array is defensively copied) and {@link #differentiate(String)} allocates
 * its own working stacks per call, so a single instance may safely be reused
 * across threads.
 */
public class SymbolicDifferentiator {

    private final Token[] rpnTokens;

    private static final int ATOMIC = 10;
    private static final int POW = 8;
    private static final int MUL_DIV = 7;
    private static final int ADD_SUB = 6;
    private static final int UNARY = 9;

    public SymbolicDifferentiator(Token[] rpnTokens) {
        if (rpnTokens == null || rpnTokens.length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        this.rpnTokens = formatTokens(rpnTokens);
    }
   public SymbolicDifferentiator(MathExpression me) {
        if (me == null || me.getCachedPostfix() == null || me.getCachedPostfix().length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        this.rpnTokens = formatTokens(me.getCachedPostfix());
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

    /**
     * Differentiates the expression with respect to {@code wrtVar}.
     *
     * @return a 2-element array: {@code [f(x), f'(x)]} as printable strings.
     */
    public String[] differentiate(String wrtVar) {
        Objects.requireNonNull(wrtVar, "wrtVar must not be null");

        final Token[] rpn = rpnTokens;
        // RPN evaluation can never need more stack slots than there are tokens.
        final Expr[] valStack = new Expr[rpn.length + 1];
        final Expr[] derStack = new Expr[rpn.length + 1];
        int sp = 0;

        for (Token t : rpn) {
            if (t.kind == Token.NUMBER) {
                valStack[sp] = new Expr(formatNumber(t.value), ATOMIC, true);
                derStack[sp] = new Expr("0", ATOMIC, true);
                sp++;
            } else if (t.kind == Token.VARIABLE) {
                boolean isWrt = (t.name != null && t.name.equals(wrtVar));
                valStack[sp] = new Expr(t.name, ATOMIC, true);
                derStack[sp] = new Expr(isWrt ? "1" : "0", ATOMIC, true);
                sp++;
            } else if (t.kind == Token.OPERATOR) {
                sp = handleOperator(t, valStack, derStack, sp);
            } else if (t.kind == Token.METHOD || t.kind == Token.FUNCTION) {
                if (t.arity == 1) {
                    sp = handleUnaryFunction(t, valStack, derStack, sp);
                } else if (t.arity == 2) {
                    sp = handleBinaryFunction(t, valStack, derStack, sp);
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

        return new String[]{valStack[0].text, derStack[0].text};
    }

    public Token[] differentiateToRPN(String wrtVar) {
        String[] result = differentiate(wrtVar);
        MathExpression derivativeExpr = new MathExpression(result[1]);
        return derivativeExpr.getCachedPostfix();
    }

    /**
     * Differentiates the expression {@code order} times with respect to
     * {@code wrtVar}, returning every intermediate order along the way.
     *
     * <p>Each step beyond the first reuses the same round-trip already used
     * by {@link #differentiateToRPN(String)}: the previous order's printed
     * derivative text is re-parsed by {@link MathExpression} back into RPN
     * tokens, and a fresh {@code SymbolicDifferentiator} differentiates that.
     * Two consequences follow directly from that, both inherent to the
     * approach rather than new risks introduced by this method:
     * <ul>
     *   <li>Any mismatch between this class's printer and
     *       {@code MathExpression}'s parser (precedence, associativity,
     *       number formatting) would compound with every additional order
     *       instead of surfacing once -- order {@code n} depends on {@code n}
     *       successful round trips, not one.</li>
     *   <li>Printed expression size can grow substantially with each order
     *       for composition-heavy inputs, since each order differentiates an
     *       already-expanded expression rather than the compact original.</li>
     * </ul>
     *
     * @param wrtVar the variable to differentiate with respect to
     * @param order  the derivative order, must be {@code >= 0}; {@code 0}
     *               returns just the original function with no
     *               differentiation performed
     * @return an {@code order + 1}-length array:
     *         {@code [f(x), f'(x), f''(x), ..., f}<sup>(order)</sup>{@code (x)]}
     */
    public String[] differentiateNTimes(String wrtVar, int order) {
        Objects.requireNonNull(wrtVar, "wrtVar must not be null");
        if (order < 0) {
            throw new IllegalArgumentException("order must be >= 0, got " + order);
        }

        String[] chain = new String[order + 1];
        if (order == 0) {
            // No differentiation needed at all; avoid the round-trip entirely.
            chain[0] = new SymbolicDifferentiator(rpnTokens).differentiate(wrtVar)[0];
            return chain;
        }

        SymbolicDifferentiator current = this;
        String[] valAndDer = current.differentiate(wrtVar);
        chain[0] = valAndDer[0];

        for (int i = 1; i <= order; i++) {
            chain[i] = valAndDer[1];
            if (i == order) {
                break;
            } 
            current = new SymbolicDifferentiator(new MathExpression(valAndDer[1]));
            valAndDer = current.differentiate(wrtVar);
        }

        return chain;
    }

    /**
     * Convenience wrapper around {@link #differentiateNTimes(String, int)}
     * that returns only the requested order's derivative text, discarding
     * the intermediate orders.
     */
    public String nthDerivative(String wrtVar, int order) {
        String[] chain = differentiateNTimes(wrtVar, order);
        return chain[chain.length - 1];
    }

    /**
     * Like {@link #differentiateToRPN(String)}, but for an arbitrary
     * derivative order: differentiates {@code order} times and returns the
     * final order's RPN tokens, e.g. for feeding into
     * {@code AutoDiffEvaluator} or a further round of symbolic
     * differentiation.
     */
    public Token[] differentiateToRPN(String wrtVar, int order) {
        String finalDerivative = nthDerivative(wrtVar, order);
        return new MathExpression(finalDerivative).getCachedPostfix();
    }

    // ====================== Expr ======================
    private static final class Expr {

        final String text;
        final int precedence;
        final boolean isAtomic;
        /**
         * Non-null if this Expr is structurally "-negationOf". Used to collapse
         * double negation correctly (-(-x) == x) without relying on fragile
         * text-prefix inspection, which breaks for compound expressions like
         * "-x + y" (only the first term is negated, not the whole thing).
         */
        final Expr negationOf;

        Expr(String text, int precedence, boolean isAtomic) {
            this(text, precedence, isAtomic, null);
        }

        Expr(String text, int precedence, boolean isAtomic, Expr negationOf) {
            this.text = text;
            this.precedence = precedence;
            this.isAtomic = isAtomic;
            this.negationOf = negationOf;
        }
    }

    private static String maybeWrap(Expr e, int parentPrec, boolean isRight) {
        if (e.isAtomic || e.precedence > parentPrec) {
            return e.text;
        }
        if (e.precedence == parentPrec) {
            return isRight ? "(" + e.text + ")" : e.text;
        }
        return "(" + e.text + ")";
    }

    // ====================== Helpers ======================
    private static boolean isZero(Expr e) {
        return e.text.equals("0") || e.text.equals("0.0");
    }

    private static boolean isOne(Expr e) {
        return e.text.equals("1") || e.text.equals("1.0");
    }

    private static boolean isNumber(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        try {
            Double.parseDouble(s);
            return true;
        } catch (NumberFormatException ignored) {
            return false;
        }
    }

    /**
     * Formats a double for round-trip printing. Deliberately avoids scientific
     * notation (e.g. "1.0E15"), since the downstream parser used to re-parse
     * printed derivatives ({@link #differentiateToRPN}) is not guaranteed to
     * accept exponent notation.
     */
    private static String formatNumber(double d) {
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            // Extremely unusual for a differentiation result; surface as-is
            // rather than silently producing an invalid literal.
            throw new ArithmeticException("Cannot format non-finite numeric result: " + d);
        }
        if (d == Math.rint(d) && Math.abs(d) < 1e15) {
            return String.valueOf((long) d);
        }
        return BigDecimal.valueOf(d).stripTrailingZeros().toPlainString();
    }

    private static Expr add(Expr a, Expr b) {
        if (isZero(a)) {
            return b;
        }
        if (isZero(b)) {
            return a;
        }
        if (isNumber(a.text) && isNumber(b.text)) {
            return new Expr(formatNumber(Double.parseDouble(a.text) + Double.parseDouble(b.text)), ATOMIC, true);
        }
        if (a.text.equals(b.text)) {
            return mul(new Expr("2", ATOMIC, true), a);
        }

        return new Expr(maybeWrap(a, ADD_SUB, false) + " + " + maybeWrap(b, ADD_SUB, true), ADD_SUB, false);
    }

    private static Expr sub(Expr a, Expr b) {
        if (isZero(b)) {
            return a;
        }
        if (isZero(a)) {
            return neg(b);
        }
        if (isNumber(a.text) && isNumber(b.text)) {
            return new Expr(formatNumber(Double.parseDouble(a.text) - Double.parseDouble(b.text)), ATOMIC, true);
        }
        if (a.text.equals(b.text)) {
            return new Expr("0", ATOMIC, true);
        }

        return new Expr(maybeWrap(a, ADD_SUB, false) + " - " + maybeWrap(b, ADD_SUB, true), ADD_SUB, false);
    }

    private static Expr mul(Expr a, Expr b) {
        if (isZero(a) || isZero(b)) {
            return new Expr("0", ATOMIC, true);
        }
        if (isOne(a)) {
            return b;
        }
        if (isOne(b)) {
            return a;
        }

        if (isNumber(a.text) && isNumber(b.text)) {
            return new Expr(formatNumber(Double.parseDouble(a.text) * Double.parseDouble(b.text)), ATOMIC, true);
        }

        if (a.text.equals("-1")) {
            return neg(b);
        }
        if (b.text.equals("-1")) {
            return neg(a);
        }

        if (isNumber(a.text) && !isNumber(b.text)) {
            return new Expr(a.text + " * " + maybeWrap(b, MUL_DIV, true), MUL_DIV, false);
        }
        if (isNumber(b.text) && !isNumber(a.text)) {
            return new Expr(b.text + " * " + maybeWrap(a, MUL_DIV, false), MUL_DIV, false);
        }

        return new Expr(maybeWrap(a, MUL_DIV, false) + " * " + maybeWrap(b, MUL_DIV, true), MUL_DIV, false);
    }

    private static Expr div(Expr a, Expr b) {
        if (isZero(a)) {
            return new Expr("0", ATOMIC, true);
        }
        if (isOne(b)) {
            return a;
        }
        if (isNumber(a.text) && isNumber(b.text)) {
            double denom = Double.parseDouble(b.text);
            if (denom == 0.0) {
                // Division by a literal zero constant: don't silently fold to
                // Infinity/NaN text that the parser can't round-trip.
                return new Expr(maybeWrap(a, MUL_DIV, false) + " / " + maybeWrap(b, MUL_DIV, true), MUL_DIV, false);
            }
            return new Expr(formatNumber(Double.parseDouble(a.text) / denom), ATOMIC, true);
        }
        if (a.text.equals(b.text)) {
            return new Expr("1", ATOMIC, true);
        }

        return new Expr(maybeWrap(a, MUL_DIV, false) + " / " + maybeWrap(b, MUL_DIV, true), MUL_DIV, false);
    }

    private static Expr pow(Expr a, Expr b) {
        if (isZero(b)) {
            return new Expr("1", ATOMIC, true);
        }
        if (isOne(b)) {
            return a;
        }
        if (isZero(a)) {
            return new Expr("0", ATOMIC, true);
        }
        if (isNumber(a.text) && isNumber(b.text)) {
            return new Expr(formatNumber(Math.pow(Double.parseDouble(a.text), Double.parseDouble(b.text))), ATOMIC, true);
        }

        // '^' is printed as a right-associative operator (standard math
        // convention). That means the BASE must be parenthesized whenever it
        // is not a single atomic token -- including when the base is itself a
        // pow expression, e.g. squaring a denominator that is already "x^3"
        // (which the quotient/power rules do routinely). maybeWrap's "leave
        // an equal-precedence operand unwrapped on the LEFT" rule is only
        // valid for left-associative operators (+, -, *, /); applying it here
        // silently corrupts the result, since a right-associative parser would
        // regroup "x^2^3" as x^(2^3)=256 instead of the intended (x^2)^3=64.
        // A negated base (or a literal negative numeric base) has the same
        // problem for a different reason: (-x)^n != -(x^n) in general, so it
        // also always needs parens rather than relying on precedence values.
        boolean baseNeedsParens = a.negationOf != null || !a.isAtomic
                || (isNumber(a.text) && a.text.startsWith("-"));
        String baseText = baseNeedsParens ? "(" + a.text + ")" : a.text;

        // The exponent side is safe under standard maybeWrap precedence
        // comparison (right-associativity means an equal-precedence pow on
        // the right is correctly left unwrapped), except for negation, which
        // gets the same defensive parenthesization as the base for clarity
        // and to avoid emitting a bare "^-y".
        String expText = (b.negationOf != null) ? "(" + b.text + ")" : maybeWrap(b, POW - 1, true);

        return new Expr(baseText + "^" + expText, POW, false);
    }

    private static Expr neg(Expr e) {
        if (isZero(e)) {
            return new Expr("0", ATOMIC, true);
        }
        // Structural double-negation collapse: -(-x) == x. This is always
        // correct because negationOf points at the exact Expr that was negated
        // to produce e, unlike text-prefix matching which cannot tell "-x + y"
        // (only the first term negated) from "-(x + y)" (whole thing negated).
        if (e.negationOf != null) {
            return e.negationOf;
        }
        if (isNumber(e.text) && e.text.startsWith("-")) {
            return new Expr(e.text.substring(1), ATOMIC, true);
        }

        String text = (e.isAtomic || e.precedence >= UNARY) ? "-" + e.text : "-(" + e.text + ")";
        return new Expr(text, UNARY, false, e);
    }

    private static Expr abs(Expr e) {
        if (isZero(e)) {
            return new Expr("0", ATOMIC, true);
        }
        return new Expr("abs(" + e.text + ")", ATOMIC, true);
    }

    // ====================== Rules ======================
    @FunctionalInterface
    public interface UnaryRule {

        Expr[] apply(Expr u, Expr du);
    }

    @FunctionalInterface
    public interface BinaryRule {

        Expr[] apply(Expr u, Expr du, Expr v, Expr dv);
    }

    // ConcurrentHashMap because registerUnaryRule/registerBinaryRule are public
    // and may be called (e.g. plugin-style rule registration) while other
    // threads are concurrently calling differentiate().
    private static final Map<String, UnaryRule> UNARY_RULES = new ConcurrentHashMap<>();
    private static final Map<String, BinaryRule> BINARY_RULES = new ConcurrentHashMap<>();

    public static void registerUnaryRule(String name, UnaryRule rule) {
        UNARY_RULES.put(name, rule);
    }

    public static void registerBinaryRule(String name, BinaryRule rule) {
        BINARY_RULES.put(name, rule);
    }

    static {
        // Unary rules (all ported)
        addUnary((u, du) -> new Expr[]{new Expr("sin(" + u.text + ")", ATOMIC, true), mul(new Expr("cos(" + u.text + ")", ATOMIC, true), du)}, Declarations.SIN);
        addUnary((u, du) -> new Expr[]{new Expr("cos(" + u.text + ")", ATOMIC, true), mul(neg(new Expr("sin(" + u.text + ")", ATOMIC, true)), du)}, Declarations.COS);
        addUnary((u, du) -> new Expr[]{new Expr("tan(" + u.text + ")", ATOMIC, true), div(du, pow(new Expr("cos(" + u.text + ")", ATOMIC, true), new Expr("2", ATOMIC, true)))}, Declarations.TAN);

        addUnary((u, du) -> new Expr[]{new Expr("asin(" + u.text + ")", ATOMIC, true), div(du, new Expr("sqrt(" + sub(new Expr("1", ATOMIC, true), pow(u, new Expr("2", ATOMIC, true))).text + ")", ATOMIC, true))}, Declarations.ARC_SIN, Declarations.ARC_SIN_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("acos(" + u.text + ")", ATOMIC, true), div(neg(du), new Expr("sqrt(" + sub(new Expr("1", ATOMIC, true), pow(u, new Expr("2", ATOMIC, true))).text + ")", ATOMIC, true))}, Declarations.ARC_COS, Declarations.ARC_COS_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("atan(" + u.text + ")", ATOMIC, true), div(du, add(new Expr("1", ATOMIC, true), pow(u, new Expr("2", ATOMIC, true))))}, Declarations.ARC_TAN, Declarations.ARC_TAN_ALT);

        addUnary((u, du) -> new Expr[]{new Expr("sec(" + u.text + ")", ATOMIC, true), mul(du, mul(new Expr("sec(" + u.text + ")", ATOMIC, true), new Expr("tan(" + u.text + ")", ATOMIC, true)))}, Declarations.SEC);
        addUnary((u, du) -> new Expr[]{new Expr("csc(" + u.text + ")", ATOMIC, true), mul(neg(du), mul(new Expr("csc(" + u.text + ")", ATOMIC, true), new Expr("cot(" + u.text + ")", ATOMIC, true)))}, "cosec", Declarations.COSEC);
        addUnary((u, du) -> new Expr[]{new Expr("cot(" + u.text + ")", ATOMIC, true), div(neg(du), pow(new Expr("sin(" + u.text + ")", ATOMIC, true), new Expr("2", ATOMIC, true)))}, Declarations.COT);

        addUnary((u, du) -> new Expr[]{new Expr("asec(" + u.text + ")", ATOMIC, true), div(du, mul(abs(u), new Expr("sqrt(" + sub(pow(u, new Expr("2", ATOMIC, true)), new Expr("1", ATOMIC, true)).text + ")", ATOMIC, true)))}, Declarations.ARC_SEC, Declarations.ARC_SEC_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("acsc(" + u.text + ")", ATOMIC, true), div(neg(du), mul(abs(u), new Expr("sqrt(" + sub(pow(u, new Expr("2", ATOMIC, true)), new Expr("1", ATOMIC, true)).text + ")", ATOMIC, true)))}, Declarations.ARC_COSEC, Declarations.ARC_COSEC_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("acot(" + u.text + ")", ATOMIC, true), div(neg(du), add(new Expr("1", ATOMIC, true), pow(u, new Expr("2", ATOMIC, true))))}, Declarations.ARC_COT, Declarations.ARC_COT_ALT);

        addUnary((u, du) -> new Expr[]{new Expr("sinh(" + u.text + ")", ATOMIC, true), mul(new Expr("cosh(" + u.text + ")", ATOMIC, true), du)}, Declarations.SINH);
        addUnary((u, du) -> new Expr[]{new Expr("cosh(" + u.text + ")", ATOMIC, true), mul(new Expr("sinh(" + u.text + ")", ATOMIC, true), du)}, Declarations.COSH);
        addUnary((u, du) -> new Expr[]{new Expr("tanh(" + u.text + ")", ATOMIC, true), mul(div(new Expr("1", ATOMIC, true), pow(new Expr("cosh(" + u.text + ")", ATOMIC, true), new Expr("2", ATOMIC, true))), du)}, Declarations.TANH);

        addUnary((u, du) -> new Expr[]{new Expr("asinh(" + u.text + ")", ATOMIC, true), div(du, new Expr("sqrt(" + add(pow(u, new Expr("2", ATOMIC, true)), new Expr("1", ATOMIC, true)).text + ")", ATOMIC, true))}, Declarations.ARC_SINH, Declarations.ARC_SINH_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("acosh(" + u.text + ")", ATOMIC, true), div(du, new Expr("sqrt(" + sub(pow(u, new Expr("2", ATOMIC, true)), new Expr("1", ATOMIC, true)).text + ")", ATOMIC, true))}, Declarations.ARC_COSH, Declarations.ARC_COSH_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("atanh(" + u.text + ")", ATOMIC, true), div(du, sub(new Expr("1", ATOMIC, true), pow(u, new Expr("2", ATOMIC, true))))}, Declarations.ARC_TANH, Declarations.ARC_TANH_ALT);

        addUnary((u, du) -> new Expr[]{new Expr("asech(" + u.text + ")", ATOMIC, true), div(neg(du), mul(u, new Expr("sqrt(" + sub(new Expr("1", ATOMIC, true), pow(u, new Expr("2", ATOMIC, true))).text + ")", ATOMIC, true)))}, Declarations.ARC_SECH, Declarations.ARC_SECH_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("acsch(" + u.text + ")", ATOMIC, true), div(neg(du), mul(abs(u), new Expr("sqrt(" + add(pow(u, new Expr("2", ATOMIC, true)), new Expr("1", ATOMIC, true)).text + ")", ATOMIC, true)))}, Declarations.ARC_COSECH, Declarations.ARC_COSECH_ALT);
        addUnary((u, du) -> new Expr[]{new Expr("acoth(" + u.text + ")", ATOMIC, true), div(neg(du), sub(pow(u, new Expr("2", ATOMIC, true)), new Expr("1", ATOMIC, true)))}, Declarations.ARC_COTH, Declarations.ARC_COTH_ALT);

        addUnary((u, du) -> new Expr[]{new Expr("sqrt(" + u.text + ")", ATOMIC, true), div(du, mul(new Expr("2", ATOMIC, true), new Expr("sqrt(" + u.text + ")", ATOMIC, true)))}, Declarations.SQRT);
        addUnary((u, du) -> new Expr[]{new Expr("cbrt(" + u.text + ")", ATOMIC, true), div(du, mul(new Expr("3", ATOMIC, true), pow(new Expr("cbrt(" + u.text + ")", ATOMIC, true), new Expr("2", ATOMIC, true))))}, Declarations.CBRT);

        addUnary((u, du) -> new Expr[]{new Expr("exp(" + u.text + ")", ATOMIC, true), mul(new Expr("exp(" + u.text + ")", ATOMIC, true), du)}, Declarations.EXP);
        addUnary((u, du) -> new Expr[]{new Expr("ln(" + u.text + ")", ATOMIC, true), div(du, u)}, Declarations.LN);
        addUnary((u, du) -> new Expr[]{new Expr("lg(" + u.text + ")", ATOMIC, true), div(du, mul(u, new Expr("ln(10)", ATOMIC, true)))}, "log", Declarations.LG);
        addUnary((u, du) -> new Expr[]{new Expr("abs(" + u.text + ")", ATOMIC, true), mul(div(u, abs(u)), du)}, "abs");

        // Binary rules
        BINARY_RULES.put(Declarations.POW, (u, du, v, dv) -> {
            Expr val = new Expr("pow(" + u.text + ", " + v.text + ")", ATOMIC, true);
            Expr der;
            if (isZero(dv)) {
                der = mul(mul(v, pow(u, sub(v, new Expr("1", ATOMIC, true)))), du);
            } else {
                Expr term1 = mul(dv, new Expr("ln(" + u.text + ")", ATOMIC, true));
                Expr term2 = mul(v, div(du, u));
                der = mul(val, add(term1, term2));
            }
            return new Expr[]{val, der};
        });

        BINARY_RULES.put(Declarations.LOG, (u, du, v, dv) -> {
            Expr lnU = new Expr("ln(" + u.text + ")", ATOMIC, true);
            Expr lnV = new Expr("ln(" + v.text + ")", ATOMIC, true);
            Expr val = div(lnU, lnV);
            Expr num = sub(mul(div(du, u), lnV), mul(lnU, div(dv, v)));
            Expr der = div(num, pow(lnV, new Expr("2", ATOMIC, true)));
            return new Expr[]{val, der};
        });

        BINARY_RULES.put("atan2", (u, du, v, dv) -> {
            Expr val = new Expr("atan2(" + u.text + ", " + v.text + ")", ATOMIC, true);
            Expr denom = add(pow(u, new Expr("2", ATOMIC, true)), pow(v, new Expr("2", ATOMIC, true)));
            Expr der = div(sub(mul(du, v), mul(u, dv)), denom);
            return new Expr[]{val, der};
        });

        BINARY_RULES.put("hypot", (u, du, v, dv) -> {
            Expr val = new Expr("hypot(" + u.text + ", " + v.text + ")", ATOMIC, true);
            Expr der = div(add(mul(u, du), mul(v, dv)), val);
            return new Expr[]{val, der};
        });
    }

    private static void addUnary(UnaryRule rule, String... names) {
        for (String name : names) {
            UNARY_RULES.put(name, rule);
        }
    }

    // ====================== Handlers ======================
    private int handleUnaryFunction(Token t, Expr[] valStack, Expr[] derStack, int sp) {
        UnaryRule rule = UNARY_RULES.get(t.name);
        if (rule == null) {
            throw new UnsupportedOperationException("Symbolic differentiation not implemented for: " + t.name);
        }
        requireOperands(sp, 1, t.name);
        Expr u = valStack[--sp];
        Expr du = derStack[sp];
        Expr[] res = rule.apply(u, du);
        valStack[sp] = res[0];
        derStack[sp] = res[1];
        return sp + 1;
    }

    private int handleBinaryFunction(Token t, Expr[] valStack, Expr[] derStack, int sp) {
        BinaryRule rule = BINARY_RULES.get(t.name);
        if (rule == null) {
            throw new UnsupportedOperationException("2-arg function not supported: " + t.name);
        }
        requireOperands(sp, 2, t.name);
        Expr v = valStack[--sp];
        Expr dv = derStack[sp];
        Expr u = valStack[--sp];
        Expr du = derStack[sp];
        Expr[] res = rule.apply(u, du, v, dv);
        valStack[sp] = res[0];
        derStack[sp] = res[1];
        return sp + 1;
    }

    private int handleOperator(Token t, Expr[] valStack, Expr[] derStack, int sp) {
        if (t.arity == 2) {
            requireOperands(sp, 2, String.valueOf(t.opChar));
            Expr b = valStack[--sp];
            Expr bDer = derStack[sp];
            Expr a = valStack[--sp];
            Expr aDer = derStack[sp];

            Expr val, der;
            switch (t.opChar) {
                case '+':
                    val = add(a, b);
                    der = add(aDer, bDer);
                    break;
                case '-':
                    val = sub(a, b);
                    der = sub(aDer, bDer);
                    break;
                case '*':
                    val = mul(a, b);
                    der = add(mul(aDer, b), mul(a, bDer));
                    break;
                case '/':
                    val = div(a, b);
                    der = div(sub(mul(aDer, b), mul(a, bDer)), pow(b, new Expr("2", ATOMIC, true)));
                    break;
                case '^':
                    val = pow(a, b);
                    if (isZero(bDer)) {
                        der = mul(mul(b, pow(a, sub(b, new Expr("1", ATOMIC, true)))), aDer);
                    } else {
                        Expr term1 = mul(bDer, new Expr("ln(" + a.text + ")", ATOMIC, true));
                        Expr term2 = mul(b, div(aDer, a));
                        der = mul(val, add(term1, term2));
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Operator not supported: " + t.opChar);
            }
            valStack[sp] = val;
            derStack[sp] = der;
            return sp + 1;
        } else if (t.arity == 1 && t.opChar == '-') {
            requireOperands(sp, 1, "unary -");
            Expr val = valStack[--sp];
            Expr der = derStack[sp];
            valStack[sp] = neg(val);
            derStack[sp] = neg(der);
            return sp + 1;
        }
        throw new UnsupportedOperationException("Unsupported operator arity");
    }

    private static void requireOperands(int sp, int needed, String opName) {
        if (sp < needed) {
            throw new IllegalStateException(
                    "Malformed RPN expression: '" + opName + "' requires " + needed
                            + " operand(s) but only " + sp + " available on the stack");
        }
    }

    public static void main(String[] args) {
        String expr = "x^3+3*x^2-5*x-8*atan2(2*x, 3)";
        MathExpression me = new MathExpression(expr);
        SymbolicDifferentiator sd = new SymbolicDifferentiator(me);
        String[] result = sd.differentiate("x");
        System.out.println("Function: " + expr);
        System.out.println("f(x)  = " + result[0]);
        System.out.println("f'(x) = " + result[1]);

        // nth-derivative example: d^3/dx^3 of the same function
        String[] chain = sd.differentiateNTimes("x", 3);
        for (int i = 0; i < chain.length; i++) {
            System.out.println("f" + "'".repeat(i) + "(x) = " + chain[i]);
        }
    }
}