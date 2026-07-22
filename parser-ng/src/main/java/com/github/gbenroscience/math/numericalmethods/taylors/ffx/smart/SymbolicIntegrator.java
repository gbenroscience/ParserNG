/*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.numericalmethods.taylors.ffx.smart;

import com.github.gbenroscience.math.numericalmethods.taylors.crx.Integrator;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.logging.Logger;

/**
 * Rule-based symbolic integrator with a numeric fallback: parses the RPN token
 * stream into a small algebraic AST ({@link Expr}), attempts a closed-form
 * antiderivative via an ordered, expandable rule book ({@link
 * IntegrationRule}), and subjects any candidate to three independent safety
 * gates before ever trusting it -- falling back transparently to a pluggable
 * {@link NumericIntegrator} otherwise.
 *
 * <h2>Reciprocal-trig and auxiliary-angle rules (new)</h2>
 * Two gaps closed in this revision:
 * <ul>
 * <li><b>{@code k/f(u)} for trig/hyperbolic {@code f}</b> (e.g.
 * {@code 2/sin(x)}, {@code 3/cos(4x)}): previously unmatched by any rule -- not
 * a polynomial denominator, not a constant log-derivative ratio, nothing for
 * u-substitution to grab onto. {@link
 *       SymbolicEngine#ruleReciprocalTrig} rewrites {@code k/sin(u)} as
 * {@code k*csc(u)} (and the five analogous pairs: cos/sec, tan/cot, sinh/csch,
 * cosh/sech, tanh/coth) and re-integrates, reusing the already-working
 * elementary table. {@code coth}, {@code sech}, {@code csch} were also added to
 * that table, {@link
 *       SymbolicEngine#funcDerivative}, and {@link SymbolicEngine#evalFunc} -- they
 * didn't exist anywhere in this engine before.</li>
 * <li><b>{@code k/(a*sin(u)+b*cos(u))}</b> (e.g. {@code 1/(sin(x)+cos(x))}):
 * needs the auxiliary-angle identity  {@code a*sin(u)+b*cos(u) = R*sin(u+phi)},
 *       {@code R=sqrt(a^2+b^2)}, {@code phi=atan2(b,a)} -- verified by hand at
 * {@code u=0} and {@code u=pi/2} with {@code a=b=1} before trusting it.
 * {@link SymbolicEngine#ruleTrigLinearCombinationReciprocal} detects this shape
 * and rewrites to {@code (k/R)*csc(u+phi)}, which the existing csc
 * antiderivative already handles -- no need to construct the tan-half-angle
 * form directly; verification checks numeric equivalence, not textual match, so
 * a differently-shaped but equal closed form is exactly as good.</li>
 * </ul>
 *
 * <h2>Algebraic cancellation</h2>
 * Two more gaps, found by tracing actual failures rather than guessing:
 * <ul>
 * <li><b>{@code 2*x*cos(x^2)}</b> (u-substitution): the correct step
 * {@code e/g'(x) = (2x*cos(x^2))/(2x) -> cos(x^2)} never reduced, because
 * {@link #simplify} had no rule for "denominator is an exact factor of the
 * numerator" -- only whole-fraction identity. Fixed by
 * {@link SymbolicEngine#tryCancelFactor}, a narrow, purely structural
 * cancellation (not general polynomial GCD): if the denominator matches the
 * whole numerator or one factor of a product numerator (searched recursively
 * through nested products), cancel it.</li>
 * <li><b>{@code exp(x)*sin(x)}</b> (self-referential integration by parts): the
 * classic closure ({@code remainder} after two applications of by-parts equals
 * {@code -1} times the original integrand) was reachable, but blocked by trying
 * a full, unconstrained recursive {@link SymbolicEngine#integrate} on the
 * remainder <em>before</em> the cheap self-referential check (which chases its
 * own tail through further {@code sin}/{@code cos}/ {@code exp} products and
 * burns the recursion-depth budget), and by {@link #simplify} lacking a "pull
 * the negative out of a product" rule needed to recognize
 * {@code exp(x)*(-sin(x))} as {@code -1*(exp(x)*sin(x))}. Fixed by trying the
 * self-referential closure <em>first</em> and pulling {@code NEG} out of
 * {@code MUL}/{@code DIV} in {@link #simplify}.</li>
 * </ul> {@link SymbolicEngine#deepEquals} was also made commutative-aware for
 * {@code MUL}/{@code ADD} (e.g. {@code cos(x)*exp(x)} and {@code exp(x)*cos(x)}
 * are the same value on different trees) as defense-in-depth against the same
 * class of matching failure in some other term ordering.
 *
 * <h2>Three safety gates, not one</h2>
 * An earlier revision of this idea relied on a single gate (differentiate the
 * candidate, spot-check against the original at 7 sample points) and, on at
 * least one run, silently declined a genuinely correct case ({@code sin(x)})
 * for reasons that couldn't be pinned down without execution -- itself evidence
 * that a single-mechanism safety net is fragile to bugs nobody has found yet.
 * This revision uses three, independent by construction:
 * <ol>
 * <li><b>Derivative verification</b>
 * ({@link SymbolicEngine#verifyAntiderivative}): differentiate the candidate
 * symbolically, compare against the original integrand at 7 points spread
 * across {@code [a,b]}. A failure here is a hard veto and can never produce a
 * false positive -- it can only cause an unnecessary (but always safe)
 * fallback.</li>
 * <li><b>Interior-singularity scan</b> ({@link #rangeLooksSuspicious}): before
 * trusting {@code F(b)-F(a)}, ask the configured fallback (if it implements
 * {@link SingularityAwareIntegrator}) whether {@code [a,b]} appears to contain
 * a pole. This exists specifically because 7 fixed sample points can miss a
 * pole that sits between them -- derivative-matching alone cannot see it, since
 * the derivative check only ever samples where it's told to look.</li>
 * <li><b>Coarse independent numeric cross-check</b> ({@link
 *       #coarseNumericCrossCheck}): a cheap, direct 32-panel composite Simpson
 * evaluation of the <em>original</em> integrand (not the antiderivative),
 * compared against the symbolic {@code F(b)-F(a)} shortfall. This is the
 * backstop: even a rule-logic bug that somehow fooled both of the above would
 * have to also produce a result close to the true numeric value to survive this
 * gate. It costs 33 function evaluations -- negligible next to the adaptive
 * engine it's protecting against having to run unnecessarily.</li>
 * </ol>
 * All three must pass. Any one failing (or throwing) falls back to the numeric
 * engine -- never partially trusts a symbolic result.
 *
 * <h2>Honesty about scope</h2>
 * General symbolic integration is undecidable; this is a real, checkable rule
 * book plus {@link SymbolicEngine#registerRule} to add more -- not a claim of
 * Risch-algorithm completeness. {@code exp(-x^2)} and general {@code x^x}
 * genuinely have no elementary closed form; this class is expected to (and
 * does, per the self-tests) fail symbolically on them and fall back cleanly.
 *
 * <h2>Pluggable numeric fallback</h2>
 * {@link Builder#fallback(NumericIntegrator)} accepts any implementation. If
 * unset, a default {@link Integrator} (Gauss-Kronrod + AD cross-check +
 * endpoint/interior singularity handling) is built lazily from the same
 * expression and variable. A fallback that also implements {@link
 * SingularityAwareIntegrator} additionally powers gate 2 above; one that
 * doesn't still works fully -- gates 1 and 3 remain in force.
 *
 * <h2>Single-variable scope, deliberately</h2>
 * The parser bails to numeric on any {@code VARIABLE} token whose name isn't
 * the integration variable. Multi-parameter symbolic calculus is a materially
 * larger feature the numeric engine already handles fully via its own
 * variable-slot mechanism, so nothing is lost by deferring it there.
 *
 * <h2>Token normalization (DRG variants)</h2> {@link #formatTokens} strips
 * {@code _rad}/{@code _grad}/{@code _deg} suffixes that the parser attaches to
 * trig function names before this engine ever sees them -- without this,
 * {@code "sin"} would arrive as {@code "sin_rad"} (or the grad/deg variant) and
 * match nothing in any rule or table below, silently forcing every trig-bearing
 * expression to the numeric fallback.
 */
public final class SymbolicIntegrator {

    /**
     * Minimal contract for "given bounds, produce a definite integral value" --
     * lets {@link SymbolicIntegrator} (or anything else) swap in any numeric
     * backend as its fallback without depending on {@link Integrator}
     * specifically.
     */
    public interface NumericIntegrator {

        double integrate(double a, double b);
    }

    /**
     * Optional capability a {@link NumericIntegrator} may offer: a cheap,
     * full-integration-free check for whether a range appears to contain a
     * singularity. {@link SymbolicIntegrator} uses this (when the configured
     * fallback provides it) as an extra safety gate before trusting a symbolic
     * shortcut across a range that might cross a pole between verification
     * sample points. Fallbacks that don't implement this remain fully usable --
     * the gate is simply skipped, and sample-based verification plus the coarse
     * numeric cross-check remain as the safety net.
     */
    public interface SingularityAwareIntegrator extends NumericIntegrator {

        boolean rangeLooksSingular(double a, double b);

    }

    private static final Logger LOG = Logger.getLogger(SymbolicIntegrator.class.getName());

    // ------------------------------------------------------------------
    // Exceptions
    // ------------------------------------------------------------------
    public static final class UnsupportedSymbolicOperationException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        public UnsupportedSymbolicOperationException(String message) {
            super(message);
        }
    }

    public static final class EvalDomainException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        public EvalDomainException(String message) {
            super(message);
        }
    }

    // ------------------------------------------------------------------
    // Expr: minimal algebraic AST
    // ------------------------------------------------------------------
    public static final class Expr {

        public enum Kind {
            CONST, VAR, ADD, SUB, MUL, DIV, NEG, POW, FUNC
        }

        public final Kind kind;
        public final double constVal;
        public final String varName;
        public final String funcName;
        public final Expr[] children;

        private static final Expr[] EMPTY = new Expr[0];

        private Expr(Kind kind, double constVal, String varName, String funcName, Expr[] children) {
            this.kind = kind;
            this.constVal = constVal;
            this.varName = varName;
            this.funcName = funcName;
            this.children = children;
        }

        public static Expr constant(double v) {
            return new Expr(Kind.CONST, v, null, null, EMPTY);
        }

        public static Expr var(String name) {
            return new Expr(Kind.VAR, 0, name, null, EMPTY);
        }

        public static Expr add(Expr a, Expr b) {
            return new Expr(Kind.ADD, 0, null, null, new Expr[]{a, b});
        }

        public static Expr sub(Expr a, Expr b) {
            return new Expr(Kind.SUB, 0, null, null, new Expr[]{a, b});
        }

        public static Expr mul(Expr a, Expr b) {
            return new Expr(Kind.MUL, 0, null, null, new Expr[]{a, b});
        }

        public static Expr div(Expr a, Expr b) {
            return new Expr(Kind.DIV, 0, null, null, new Expr[]{a, b});
        }

        public static Expr neg(Expr a) {
            return new Expr(Kind.NEG, 0, null, null, new Expr[]{a});
        }

        public static Expr pow(Expr a, Expr b) {
            return new Expr(Kind.POW, 0, null, null, new Expr[]{a, b});
        }

        public static Expr func(String name, Expr... args) {
            return new Expr(Kind.FUNC, 0, null, name, args);
        }

        @Override
        public String toString() {
            switch (kind) {
                case CONST:
                    return trimNum(constVal);
                case VAR:
                    return varName;
                case ADD:
                    return "(" + children[0] + " + " + children[1] + ")";
                case SUB:
                    return "(" + children[0] + " - " + children[1] + ")";
                case MUL:
                    return "(" + children[0] + " * " + children[1] + ")";
                case DIV:
                    return "(" + children[0] + " / " + children[1] + ")";
                case NEG:
                    return "(-" + children[0] + ")";
                case POW:
                    return "(" + children[0] + "^" + children[1] + ")";
                case FUNC: {
                    StringBuilder sb = new StringBuilder(funcName).append('(');
                    for (int i = 0; i < children.length; i++) {
                        if (i > 0) {
                            sb.append(", ");
                        }
                        sb.append(children[i]);
                    }
                    return sb.append(')').toString();
                }
                default:
                    return "?";
            }
        }

        private static String trimNum(double v) {
            if (v == Math.floor(v) && !Double.isInfinite(v) && Math.abs(v) < 1e15) {
                return Long.toString((long) v);
            }
            return Double.toString(v);
        }
    }

    // ------------------------------------------------------------------
    // IntegrationRule: the expandable rule-book interface
    // ------------------------------------------------------------------
    public interface IntegrationRule {

        /**
         * Returns a (not-yet-simplified) candidate antiderivative, or null if
         * this rule does not apply.
         *
         * @param integrand
         * @param var
         * @param engine
         * @param depth
         * @return
         */
        Expr tryIntegrate(Expr integrand, String var, SymbolicEngine engine, int depth);
    }

    // ------------------------------------------------------------------
    // SymbolicEngine: parse / simplify / differentiate / integrate / evaluate
    // ------------------------------------------------------------------
    public static final class SymbolicEngine {

        private static final int MAX_INTEGRATE_DEPTH = 14;
        private static final int MAX_BY_PARTS_DEPTH = 4;
        private static final int MAX_POLY_DEGREE = 12;
        private static final String DUMMY_VAR = "__u__";

        private final List<IntegrationRule> rules = new ArrayList<>();

        public SymbolicEngine() {
            registerDefaultRules();
        }

        public void registerRule(IntegrationRule rule) {
            rules.add(rule);
        }

        public void registerRuleFirst(IntegrationRule rule) {
            rules.add(0, rule);
        }

        // -------------------- parsing --------------------
        public Expr parseToExpr(Token[] rpn, String wrtVar) {
            Deque<Expr> stack = new ArrayDeque<>();
            for (Token t : rpn) {
                switch (t.kind) {
                    case Token.NUMBER:
                        stack.push(Expr.constant(t.value));
                        break;
                    case Token.VARIABLE:
                        if (t.name == null || !t.name.equals(wrtVar)) {
                            throw new UnsupportedSymbolicOperationException(
                                    "symbolic engine only supports a single variable ('" + wrtVar
                                    + "'); found '" + t.name + "'");
                        }
                        stack.push(Expr.var(wrtVar));
                        break;
                    case Token.OPERATOR:
                        parseOperator(t, stack);
                        break;
                    case Token.FUNCTION:
                    case Token.METHOD:
                        if (t.arity == 1) {
                            Expr a = requirePop(stack);
                            if ("pow".equals(t.name)) {
                                throw new UnsupportedSymbolicOperationException("unary 'pow' unexpected");
                            }

                            stack.push(Expr.func(t.name, a));
                        } else if (t.arity == 2) {
                            Expr b = requirePop(stack);
                            Expr a = requirePop(stack);
                            if ("pow".equals(t.name)) {
                                stack.push(Expr.pow(a, b));
                            } else {
                                stack.push(Expr.func(t.name, a, b));
                            }
                        } else {
                            throw new UnsupportedSymbolicOperationException(
                                    "unsupported arity " + t.arity + " for " + t.name);
                        }
                        break;
                    case Token.LPAREN:
                    case Token.RPAREN:
                    case Token.COMMA:
                        // Structural no-ops in this postfix stream (mirrors how every
                        // evaluateRPN loop elsewhere in this codebase silently skips
                        // token kinds it doesn't explicitly handle).
                        break;
                    default:
                        throw new UnsupportedSymbolicOperationException(
                                "unsupported token kind for symbolic parsing: " + Token.getKind(t.kind));
                }
            }
            if (stack.size() != 1) {
                throw new UnsupportedSymbolicOperationException(
                        "malformed RPN for symbolic parsing: expected 1 result, got " + stack.size());
            }
            return stack.pop();
        }

        private static Expr requirePop(Deque<Expr> stack) {
            if (stack.isEmpty()) {
                throw new UnsupportedSymbolicOperationException("malformed RPN: stack underflow");
            }
            return stack.pop();
        }

        private void parseOperator(Token t, Deque<Expr> stack) {
            if (t.arity == 2) {
                Expr b = requirePop(stack);
                Expr a = requirePop(stack);
                switch (t.opChar) {
                    case '+':
                        stack.push(Expr.add(a, b));
                        break;
                    case '-':
                        stack.push(Expr.sub(a, b));
                        break;
                    case '*':
                        stack.push(Expr.mul(a, b));
                        break;
                    case '/':
                        stack.push(Expr.div(a, b));
                        break;
                    case '^':
                        stack.push(Expr.pow(a, b));
                        break;
                    default:
                        throw new UnsupportedSymbolicOperationException("unsupported binary operator '" + t.opChar + "'");
                }
            } else {
                Expr a = requirePop(stack);
                switch (t.opChar) {
                    case '-':
                        stack.push(Expr.neg(a));
                        break;
                    case '²':
                        stack.push(Expr.pow(a, Expr.constant(2)));
                        break;
                    case '³':
                        stack.push(Expr.pow(a, Expr.constant(3)));
                        break;
                    case '√':
                        stack.push(Expr.func("sqrt", a));
                        break;
                    default:
                        if (t.opChar == Token.CBRT_DEF) {
                            stack.push(Expr.func("cbrt", a));
                        } else if (t.opChar == Token.INVERSE_DEF) {
                            stack.push(Expr.div(Expr.constant(1), a));
                        } else {
                            throw new UnsupportedSymbolicOperationException("unsupported unary operator '" + t.opChar + "'");
                        }
                }
            }
        }

        // -------------------- structural helpers --------------------
        public boolean containsVar(Expr e, String var) {
            if (e.kind == Expr.Kind.VAR) {
                return e.varName.equals(var);
            }
            for (Expr c : e.children) {
                if (containsVar(c, var)) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Structural equality, commutative-aware for {@code ADD}/{@code MUL}
         * (e.g. {@code cos(x)*exp(x)} and {@code exp(x)*cos(x)} are the same
         * value on different trees). Both orientations are checked before
         * declaring a mismatch.
         */
        public boolean deepEquals(Expr a, Expr b) {
            if (a.kind != b.kind) {
                return false;
            }
            switch (a.kind) {
                case CONST:
                    return a.constVal == b.constVal;
                case VAR:
                    return a.varName.equals(b.varName);
                case FUNC:
                    if (!a.funcName.equals(b.funcName) || a.children.length != b.children.length) {
                        return false;
                    }
                    for (int i = 0; i < a.children.length; i++) {
                        if (!deepEquals(a.children[i], b.children[i])) {
                            return false;
                        }
                    }
                    return true;
                case ADD:
                case MUL:
                    return (deepEquals(a.children[0], b.children[0]) && deepEquals(a.children[1], b.children[1]))
                            || (deepEquals(a.children[0], b.children[1]) && deepEquals(a.children[1], b.children[0]));
                default:
                    break;
            }
            if (a.children.length != b.children.length) {
                return false;
            }
            for (int i = 0; i < a.children.length; i++) {
                if (!deepEquals(a.children[i], b.children[i])) {
                    return false;
                }
            }
            return true;
        }

        public Expr replaceSubexpr(Expr e, Expr target, Expr replacement) {
            if (deepEquals(e, target)) {
                return replacement;
            }
            if (e.children.length == 0) {
                return e;
            }
            Expr[] nc = new Expr[e.children.length];
            boolean changed = false;
            for (int i = 0; i < e.children.length; i++) {
                nc[i] = replaceSubexpr(e.children[i], target, replacement);
                if (nc[i] != e.children[i]) {
                    changed = true;
                }
            }
            if (!changed) {
                return e;
            }
            switch (e.kind) {
                case ADD:
                    return Expr.add(nc[0], nc[1]);
                case SUB:
                    return Expr.sub(nc[0], nc[1]);
                case MUL:
                    return Expr.mul(nc[0], nc[1]);
                case DIV:
                    return Expr.div(nc[0], nc[1]);
                case NEG:
                    return Expr.neg(nc[0]);
                case POW:
                    return Expr.pow(nc[0], nc[1]);
                case FUNC:
                    return Expr.func(e.funcName, nc);
                default:
                    return e;
            }
        }

        // -------------------- simplification --------------------
        public Expr simplify(Expr e) {
            if (e.children.length == 0) {
                return e;
            }
            Expr[] sc = new Expr[e.children.length];
            for (int i = 0; i < sc.length; i++) {
                sc[i] = simplify(e.children[i]);
            }

            switch (e.kind) {
                case ADD: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) {
                        return Expr.constant(a.constVal + b.constVal);
                    }
                    if (a.kind == Expr.Kind.CONST && a.constVal == 0.0) {
                        return b;
                    }
                    if (b.kind == Expr.Kind.CONST && b.constVal == 0.0) {
                        return a;
                    }
                    return Expr.add(a, b);
                }
                case SUB: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) {
                        return Expr.constant(a.constVal - b.constVal);
                    }
                    if (b.kind == Expr.Kind.CONST && b.constVal == 0.0) {
                        return a;
                    }
                    if (deepEquals(a, b)) {
                        return Expr.constant(0);
                    }
                    if (a.kind == Expr.Kind.CONST && a.constVal == 0.0) {
                        return simplify(Expr.neg(b));
                    }
                    return Expr.sub(a, b);
                }
                case MUL: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) {
                        return Expr.constant(a.constVal * b.constVal);
                    }
                    if ((a.kind == Expr.Kind.CONST && a.constVal == 0.0) || (b.kind == Expr.Kind.CONST && b.constVal == 0.0)) {
                        return Expr.constant(0);
                    }
                    if (a.kind == Expr.Kind.CONST && a.constVal == 1.0) {
                        return b;
                    }
                    if (b.kind == Expr.Kind.CONST && b.constVal == 1.0) {
                        return a;
                    }
                    // Pull a negation out of a product: a*(-b) = -(a*b). Needed so that e.g.
                    // exp(x)*(-sin(x)) reduces to a form directly comparable (via deepEquals,
                    // inside a division, to fold to a literal constant ratio) against the
                    // original integrand exp(x)*sin(x) in the self-referential by-parts check.
                    if (a.kind == Expr.Kind.NEG) {
                        return simplify(Expr.neg(Expr.mul(a.children[0], b)));
                    }
                    if (b.kind == Expr.Kind.NEG) {
                        return simplify(Expr.neg(Expr.mul(a, b.children[0])));
                    }
                    return Expr.mul(a, b);
                }
                case DIV: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST && b.constVal != 0.0) {
                        return Expr.constant(a.constVal / b.constVal);
                    }
                    if (b.kind == Expr.Kind.CONST && b.constVal == 1.0) {
                        return a;
                    }
                    if (a.kind == Expr.Kind.CONST && a.constVal == 0.0) {
                        return Expr.constant(0);
                    }
                    // Same NEG-pulling identity as MUL, applied to division: (-a)/b = -(a/b),
                    // a/(-b) = -(a/b).
                    if (a.kind == Expr.Kind.NEG) {
                        return simplify(Expr.neg(Expr.div(a.children[0], b)));
                    }
                    if (b.kind == Expr.Kind.NEG) {
                        return simplify(Expr.neg(Expr.div(a, b.children[0])));
                    }
                    if (deepEquals(a, b)) {
                        return Expr.constant(1);
                    }
                    // Structural (not general polynomial-GCD) cancellation: denominator matches
                    // the whole numerator or one factor of a product numerator. This is exactly
                    // what u-substitution's e/g'(x) step needs -- e.g.
                    // (2x*cos(x^2))/(2x) -> cos(x^2) -- and what lets the self-referential
                    // by-parts ratio check recognize exp(x)*(-sin(x)) / (exp(x)*sin(x)) = -1.
                    Expr cancelled = tryCancelFactor(a, b);
                    if (cancelled != null) {
                        return cancelled;
                    }
                    return Expr.div(a, b);
                }
                case NEG: {
                    Expr a = sc[0];
                    if (a.kind == Expr.Kind.CONST) {
                        return Expr.constant(-a.constVal);
                    }
                    if (a.kind == Expr.Kind.NEG) {
                        return a.children[0];
                    }
                    return Expr.neg(a);
                }
                case POW: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) {
                        double r = Math.pow(a.constVal, b.constVal);
                        if (Double.isFinite(r)) {
                            return Expr.constant(r);
                        }
                    }
                    if (b.kind == Expr.Kind.CONST && b.constVal == 1.0) {
                        return a;
                    }
                    if (b.kind == Expr.Kind.CONST && b.constVal == 0.0) {
                        return Expr.constant(1);
                    }
                    if (a.kind == Expr.Kind.POW && b.kind == Expr.Kind.CONST && a.children[1].kind == Expr.Kind.CONST) {
                        return simplify(Expr.pow(a.children[0], Expr.constant(a.children[1].constVal * b.constVal)));
                    }
                    return Expr.pow(a, b);
                }
                case FUNC: {
                    if (sc.length == 1 && sc[0].kind == Expr.Kind.CONST) {
                        Double folded = tryFoldUnary(e.funcName, sc[0].constVal);
                        if (folded != null) {
                            return Expr.constant(folded);
                        }
                    }
                    return Expr.func(e.funcName, sc);
                }
                default:
                    return e;
            }
        }

        /**
         * If {@code denom} matches {@code numerator} exactly, or matches one
         * whole multiplicative factor of {@code numerator} (searched
         * recursively through nested products), returns the remaining factor(s)
         * with that factor removed; otherwise null. Narrow and purely
         * structural -- not general polynomial GCD -- but it is exactly what
         * u-substitution's {@code e/g'(x)} step and the integration-by-parts
         * self-referential ratio check both need.
         */
        private Expr tryCancelFactor(Expr numerator, Expr denom) {
            if (deepEquals(numerator, denom)) {
                return Expr.constant(1);
            }
            if (numerator.kind == Expr.Kind.MUL) {
                Expr a = numerator.children[0], b = numerator.children[1];
                if (deepEquals(a, denom)) {
                    return b;
                }
                if (deepEquals(b, denom)) {
                    return a;
                }
                Expr ra = tryCancelFactor(a, denom);
                if (ra != null) {
                    return simplify(Expr.mul(ra, b));
                }
                Expr rb = tryCancelFactor(b, denom);
                if (rb != null) {
                    return simplify(Expr.mul(a, rb));
                }
            }
            return null;
        }

        private Double tryFoldUnary(String name, double x) {
            try {
                switch (name) {
                    case "sin":
                        return Math.sin(x);
                    case "cos":
                        return Math.cos(x);
                    case "exp":
                        return Math.exp(x);
                    case "sqrt":
                        return x >= 0 ? Math.sqrt(x) : null;
                    case "abs":
                        return Math.abs(x);
                    case "ln":
                        return x > 0 ? Math.log(x) : null;
                    default:
                        return null;
                }
            } catch (RuntimeException ignore) {
                return null;
            }
        }

        // -------------------- differentiation --------------------
        public Expr diff(Expr e, String var) {
            switch (e.kind) {
                case CONST:
                    return Expr.constant(0);
                case VAR:
                    return Expr.constant(e.varName.equals(var) ? 1.0 : 0.0);
                case ADD:
                    return simplify(Expr.add(diff(e.children[0], var), diff(e.children[1], var)));
                case SUB:
                    return simplify(Expr.sub(diff(e.children[0], var), diff(e.children[1], var)));
                case NEG:
                    return simplify(Expr.neg(diff(e.children[0], var)));
                case MUL: {
                    Expr a = e.children[0], b = e.children[1];
                    Expr da = diff(a, var), db = diff(b, var);
                    return simplify(Expr.add(Expr.mul(da, b), Expr.mul(a, db)));
                }
                case DIV: {
                    Expr a = e.children[0], b = e.children[1];
                    Expr da = diff(a, var), db = diff(b, var);
                    return simplify(Expr.div(Expr.sub(Expr.mul(da, b), Expr.mul(a, db)), Expr.pow(b, Expr.constant(2))));
                }
                case POW: {
                    Expr base = e.children[0], expo = e.children[1];
                    boolean expoConst = !containsVar(expo, var);
                    boolean baseConst = !containsVar(base, var);
                    if (expoConst && baseConst) {
                        return Expr.constant(0);
                    }
                    if (expoConst) {
                        Expr newExp = simplify(Expr.sub(expo, Expr.constant(1)));
                        Expr dbase = diff(base, var);
                        return simplify(Expr.mul(Expr.mul(expo, Expr.pow(base, newExp)), dbase));
                    }
                    if (baseConst) {
                        Expr dexpo = diff(expo, var);
                        return simplify(Expr.mul(Expr.mul(Expr.pow(base, expo), Expr.func("ln", base)), dexpo));
                    }
                    Expr dbase = diff(base, var);
                    Expr dexpo = diff(expo, var);
                    Expr inner = Expr.add(Expr.mul(dexpo, Expr.func("ln", base)), Expr.div(Expr.mul(expo, dbase), base));
                    return simplify(Expr.mul(Expr.pow(base, expo), inner));
                }
                case FUNC: {
                    if (e.children.length == 1) {
                        Expr arg = e.children[0];
                        Expr darg = diff(arg, var);
                        Expr outer = funcDerivative(e.funcName, arg);
                        if (outer == null) {
                            throw new UnsupportedSymbolicOperationException("no known derivative for " + e.funcName);
                        }
                        return simplify(Expr.mul(outer, darg));
                    } else if (e.children.length == 2 && "atan2".equals(e.funcName)) {
                        Expr u = e.children[0], v = e.children[1];
                        Expr du = diff(u, var), dv = diff(v, var);
                        Expr denom = Expr.add(Expr.pow(u, Expr.constant(2)), Expr.pow(v, Expr.constant(2)));
                        Expr numer = Expr.sub(Expr.mul(v, du), Expr.mul(u, dv));
                        return simplify(Expr.div(numer, denom));
                    } else {
                        throw new UnsupportedSymbolicOperationException(
                                "no known derivative for " + e.funcName + "/" + e.children.length + " args");
                    }
                }
                default:
                    throw new IllegalStateException("unreachable");
            }
        }

        private Expr funcDerivative(String name, Expr arg) {
            switch (name) {
                case "sin":
                    return Expr.func("cos", arg);
                case "cos":
                    return Expr.neg(Expr.func("sin", arg));
                case "tan":
                    return Expr.div(Expr.constant(1), Expr.pow(Expr.func("cos", arg), Expr.constant(2)));
                case "sec":
                    return Expr.div(Expr.func("sin", arg), Expr.pow(Expr.func("cos", arg), Expr.constant(2)));
                case "cosec":
                case "csc":
                    return Expr.neg(Expr.div(Expr.func("cos", arg), Expr.pow(Expr.func("sin", arg), Expr.constant(2))));
                case "cot":
                    return Expr.neg(Expr.div(Expr.constant(1), Expr.pow(Expr.func("sin", arg), Expr.constant(2))));
                case "exp":
                    return Expr.func("exp", arg);
                case "ln":
                    return Expr.div(Expr.constant(1), arg);
                case "lg":
                    return Expr.div(Expr.constant(1), Expr.mul(arg, Expr.constant(Math.log(10))));
                case "sqrt":
                    return Expr.div(Expr.constant(1), Expr.mul(Expr.constant(2), Expr.func("sqrt", arg)));
                case "cbrt":
                    return Expr.div(Expr.constant(1), Expr.mul(Expr.constant(3), Expr.pow(Expr.func("cbrt", arg), Expr.constant(2))));
                case "sinh":
                    return Expr.func("cosh", arg);
                case "cosh":
                    return Expr.func("sinh", arg);
                case "tanh":
                    return Expr.div(Expr.constant(1), Expr.pow(Expr.func("cosh", arg), Expr.constant(2)));
                case "coth":
                    return Expr.neg(Expr.div(Expr.constant(1), Expr.pow(Expr.func("sinh", arg), Expr.constant(2))));
                case "sech":
                    return Expr.neg(Expr.mul(Expr.func("sech", arg), Expr.func("tanh", arg)));
                case "csch":
                case "cosech":
                    return Expr.neg(Expr.mul(Expr.func("csch", arg), Expr.func("coth", arg)));
                case "asin":
                case "arcsin":
                    return Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(arg, Expr.constant(2)))));
                case "acos":
                case "arccos":
                    return Expr.neg(Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(arg, Expr.constant(2))))));
                case "atan":
                case "arctan":
                    return Expr.div(Expr.constant(1), Expr.add(Expr.constant(1), Expr.pow(arg, Expr.constant(2))));
                case "asinh":
                case "arcsinh":
                    return Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.add(Expr.pow(arg, Expr.constant(2)), Expr.constant(1))));
                case "acosh":
                case "arccosh":
                    return Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.sub(Expr.pow(arg, Expr.constant(2)), Expr.constant(1))));
                case "atanh":
                case "arctanh":
                    return Expr.div(Expr.constant(1), Expr.sub(Expr.constant(1), Expr.pow(arg, Expr.constant(2))));
                case "abs":
                    return Expr.div(arg, Expr.func("abs", arg));
                default:
                    return null;
            }
        }

        // -------------------- integration --------------------
        public Expr integrate(Expr integrand, String var) {
            return integrate(integrand, var, 0);
        }

        Expr integrate(Expr integrand, String var, int depth) {
            if (depth > MAX_INTEGRATE_DEPTH) {
                return null;
            }
            Expr e = simplify(integrand);
            for (IntegrationRule rule : rules) {
                Expr result = rule.tryIntegrate(e, var, this, depth);
                if (result != null) {
                    return simplify(result);
                }
            }
            return null;
        }

        // -------------------- numeric evaluation --------------------
        public double numericEval(Expr e, String var, double x) {
            switch (e.kind) {
                case CONST:
                    return e.constVal;
                case VAR:
                    return e.varName.equals(var) ? x : 0.0;
                case ADD:
                    return numericEval(e.children[0], var, x) + numericEval(e.children[1], var, x);
                case SUB:
                    return numericEval(e.children[0], var, x) - numericEval(e.children[1], var, x);
                case MUL:
                    return numericEval(e.children[0], var, x) * numericEval(e.children[1], var, x);
                case DIV: {
                    double denom = numericEval(e.children[1], var, x);
                    if (Math.abs(denom) < 1e-300) {
                        throw new EvalDomainException("division by zero");
                    }
                    return numericEval(e.children[0], var, x) / denom;
                }
                case NEG:
                    return -numericEval(e.children[0], var, x);
                case POW: {
                    double base = numericEval(e.children[0], var, x);
                    double expo = numericEval(e.children[1], var, x);
                    double r = Math.pow(base, expo);
                    if (!Double.isFinite(r) && Double.isFinite(base) && Double.isFinite(expo)) {
                        throw new EvalDomainException("pow domain error: " + base + "^" + expo);
                    }
                    return r;
                }
                case FUNC:
                    return evalFunc(e.funcName, e.children, var, x);
                default:
                    throw new IllegalStateException("unreachable");
            }
        }

        private double evalFunc(String name, Expr[] children, String var, double x) {
            if (children.length == 2 && "atan2".equals(name)) {
                return Math.atan2(numericEval(children[0], var, x), numericEval(children[1], var, x));
            }
            double u = numericEval(children[0], var, x);
            switch (name) {
                case "sin":
                    return Math.sin(u);
                case "cos":
                    return Math.cos(u);
                case "tan":
                    return Math.tan(u);
                case "sec":
                    return 1.0 / Math.cos(u);
                case "cosec":
                case "csc":
                    return 1.0 / Math.sin(u);
                case "cot":
                    return 1.0 / Math.tan(u);
                case "exp":
                    return Math.exp(u);
                case "ln":
                    if (u <= 0) {
                        throw new EvalDomainException("ln domain");
                    }
                    return Math.log(u);
                case "lg":
                    if (u <= 0) {
                        throw new EvalDomainException("lg domain");
                    }
                    return Math.log10(u);
                case "sqrt":
                    if (u < 0) {
                        throw new EvalDomainException("sqrt domain");
                    }
                    return Math.sqrt(u);
                case "cbrt":
                    return Math.cbrt(u);
                case "sinh":
                    return Math.sinh(u);
                case "cosh":
                    return Math.cosh(u);
                case "tanh":
                    return Math.tanh(u);
                case "coth":
                    return Math.cosh(u) / Math.sinh(u);
                case "sech":
                    return 1.0 / Math.cosh(u);
                case "csch":
                case "cosech":
                    return 1.0 / Math.sinh(u);
                case "asin":
                case "arcsin":
                    if (Math.abs(u) > 1) {
                        throw new EvalDomainException("asin domain");
                    }
                    return Math.asin(u);
                case "acos":
                case "arccos":
                    if (Math.abs(u) > 1) {
                        throw new EvalDomainException("acos domain");
                    }
                    return Math.acos(u);
                case "atan":
                case "arctan":
                    return Math.atan(u);
                case "asinh":
                case "arcsinh":
                    return Math.log(u + Math.sqrt(u * u + 1));
                case "acosh":
                case "arccosh":
                    if (u < 1) {
                        throw new EvalDomainException("acosh domain");
                    }
                    return Math.log(u + Math.sqrt(u * u - 1));
                case "atanh":
                case "arctanh":
                    if (Math.abs(u) >= 1) {
                        throw new EvalDomainException("atanh domain");
                    }
                    return 0.5 * Math.log((1 + u) / (1 - u));
                case "abs":
                    return Math.abs(u);
                default:
                    throw new EvalDomainException("unknown function for symbolic evaluation: " + name);
            }
        }

        // -------------------- verification (gate 1) --------------------
        public boolean verifyAntiderivative(Expr integrand, Expr antiderivative, String var, double a, double b) {
            Expr derivative;
            try {
                derivative = simplify(diff(antiderivative, var));
            } catch (RuntimeException e) {
                return false;
            }
            double lo = Math.min(a, b), hi = Math.max(a, b);
            double width = hi - lo;
            if (width <= 0) {
                return true;
            }
            int samples = 7;
            int checked = 0;
            for (int i = 0; i < samples; i++) {
                double frac = (i + 0.5) / samples;
                double x = lo + frac * width;
                double expected, actual;
                try {
                    expected = numericEval(integrand, var, x);
                    actual = numericEval(derivative, var, x);
                } catch (EvalDomainException skip) {
                    continue;
                }
                if (!Double.isFinite(expected) || !Double.isFinite(actual)) {
                    continue;
                }
                double scale = Math.max(1.0, Math.max(Math.abs(expected), Math.abs(actual)));
                if (Math.abs(expected - actual) > 1e-6 * scale) {
                    return false;
                }
                checked++;
            }
            return checked > 0;
        }

        // -------------------- affine / polynomial coefficient extraction --------------------
        double[] affineCoeffs(Expr e, String var) {
            switch (e.kind) {
                case CONST:
                    return new double[]{0.0, e.constVal};
                case VAR:
                    return new double[]{1.0, 0.0};
                case NEG: {
                    double[] c = affineCoeffs(e.children[0], var);
                    return c == null ? null : new double[]{-c[0], -c[1]};
                }
                case ADD: {
                    double[] ca = affineCoeffs(e.children[0], var), cb = affineCoeffs(e.children[1], var);
                    return (ca == null || cb == null) ? null : new double[]{ca[0] + cb[0], ca[1] + cb[1]};
                }
                case SUB: {
                    double[] ca = affineCoeffs(e.children[0], var), cb = affineCoeffs(e.children[1], var);
                    return (ca == null || cb == null) ? null : new double[]{ca[0] - cb[0], ca[1] - cb[1]};
                }
                case MUL: {
                    double[] ca = affineCoeffs(e.children[0], var), cb = affineCoeffs(e.children[1], var);
                    if (ca == null || cb == null) {
                        return null;
                    }
                    if (ca[0] == 0.0) {
                        return new double[]{ca[1] * cb[0], ca[1] * cb[1]};
                    }
                    if (cb[0] == 0.0) {
                        return new double[]{cb[1] * ca[0], cb[1] * ca[1]};
                    }
                    return null;
                }
                case DIV: {
                    double[] cb = affineCoeffs(e.children[1], var);
                    if (cb == null || cb[0] != 0.0 || cb[1] == 0.0) {
                        return null;
                    }
                    double[] ca = affineCoeffs(e.children[0], var);
                    return ca == null ? null : new double[]{ca[0] / cb[1], ca[1] / cb[1]};
                }
                case POW: {
                    if (e.children[1].kind == Expr.Kind.CONST) {
                        if (e.children[1].constVal == 0.0) {
                            return new double[]{0.0, 1.0};
                        }
                        if (e.children[1].constVal == 1.0) {
                            return affineCoeffs(e.children[0], var);
                        }
                    }
                    return null;
                }
                default:
                    return null;
            }
        }

        double[] polyCoeffs(Expr e, String var, int maxDegree) {
            switch (e.kind) {
                case CONST:
                    return new double[]{e.constVal};
                case VAR:
                    return new double[]{0.0, 1.0};
                case NEG: {
                    double[] c = polyCoeffs(e.children[0], var, maxDegree);
                    if (c == null) {
                        return null;
                    }
                    double[] r = c.clone();
                    for (int i = 0; i < r.length; i++) {
                        r[i] = -r[i];
                    }
                    return r;
                }
                case ADD:
                    return addPoly(polyCoeffs(e.children[0], var, maxDegree), polyCoeffs(e.children[1], var, maxDegree), maxDegree);
                case SUB: {
                    double[] cb = polyCoeffs(e.children[1], var, maxDegree);
                    if (cb == null) {
                        return null;
                    }
                    double[] negb = cb.clone();
                    for (int i = 0; i < negb.length; i++) {
                        negb[i] = -negb[i];
                    }
                    return addPoly(polyCoeffs(e.children[0], var, maxDegree), negb, maxDegree);
                }
                case MUL:
                    return mulPoly(polyCoeffs(e.children[0], var, maxDegree), polyCoeffs(e.children[1], var, maxDegree), maxDegree);
                case POW: {
                    if (e.children[1].kind == Expr.Kind.CONST) {
                        double nv = e.children[1].constVal;
                        if (nv == Math.floor(nv) && nv >= 0 && nv <= maxDegree) {
                            int n = (int) nv;
                            double[] base = polyCoeffs(e.children[0], var, maxDegree);
                            if (base == null) {
                                return null;
                            }
                            double[] result = new double[]{1.0};
                            for (int i = 0; i < n; i++) {
                                result = mulPoly(result, base, maxDegree);
                                if (result == null) {
                                    return null;
                                }
                            }
                            return result;
                        }
                    }
                    return null;
                }
                default:
                    return null;
            }
        }

        private static double[] addPoly(double[] a, double[] b, int maxDegree) {
            if (a == null || b == null) {
                return null;
            }
            int len = Math.max(a.length, b.length);
            if (len - 1 > maxDegree) {
                return null;
            }
            double[] r = new double[len];
            for (int i = 0; i < len; i++) {
                r[i] = (i < a.length ? a[i] : 0) + (i < b.length ? b[i] : 0);
            }
            return r;
        }

        private static double[] mulPoly(double[] a, double[] b, int maxDegree) {
            if (a == null || b == null) {
                return null;
            }
            int len = a.length + b.length - 1;
            if (len < 1) {
                return new double[]{0.0};
            }
            if (len - 1 > maxDegree) {
                return null;
            }
            double[] r = new double[len];
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < b.length; j++) {
                    r[i + j] += a[i] * b[j];
                }
            }
            return r;
        }

        private Expr polyToExpr(double[] coeffs, String var) {
            Expr result = Expr.constant(0);
            for (int i = coeffs.length - 1; i >= 0; i--) {
                if (coeffs[i] == 0.0) {
                    continue;
                }
                Expr term = (i == 0) ? Expr.constant(coeffs[i])
                        : Expr.mul(Expr.constant(coeffs[i]), Expr.pow(Expr.var(var), Expr.constant(i)));
                result = Expr.add(result, term);
            }
            return simplify(result);
        }

        Double constantRatio(Expr a, Expr b) {
            Expr ratio = simplify(Expr.div(a, b));
            return ratio.kind == Expr.Kind.CONST ? ratio.constVal : null;
        }

        // -------------------- elementary antiderivative table --------------------
        Expr elementaryAntiderivative(String name, Expr u) {
            switch (name) {
                case "sin":
                    return Expr.neg(Expr.func("cos", u));
                case "cos":
                    return Expr.func("sin", u);
                case "tan":
                    return Expr.neg(Expr.func("ln", Expr.func("abs", Expr.func("cos", u))));
                case "sec":
                    return Expr.func("ln", Expr.func("abs", Expr.add(Expr.func("sec", u), Expr.func("tan", u))));
                case "cosec":
                case "csc":
                    return Expr.neg(Expr.func("ln", Expr.func("abs", Expr.add(Expr.func("cosec", u), Expr.func("cot", u)))));
                case "cot":
                    return Expr.func("ln", Expr.func("abs", Expr.func("sin", u)));
                case "exp":
                    return Expr.func("exp", u);
                case "ln":
                    return Expr.sub(Expr.mul(u, Expr.func("ln", u)), u);
                case "lg":
                    return Expr.div(Expr.sub(Expr.mul(u, Expr.func("ln", u)), u), Expr.constant(Math.log(10)));
                case "sqrt":
                    return Expr.mul(Expr.constant(2.0 / 3.0), Expr.pow(u, Expr.constant(1.5)));
                case "cbrt":
                    return Expr.mul(Expr.constant(0.75), Expr.pow(u, Expr.constant(4.0 / 3.0)));
                case "sinh":
                    return Expr.func("cosh", u);
                case "cosh":
                    return Expr.func("sinh", u);
                case "tanh":
                    return Expr.func("ln", Expr.func("cosh", u));
                case "coth":
                    return Expr.func("ln", Expr.func("abs", Expr.func("sinh", u)));
                case "sech":
                    return Expr.func("atan", Expr.func("sinh", u));
                case "csch":
                case "cosech":
                    return Expr.func("ln", Expr.func("abs", Expr.func("tanh", Expr.div(u, Expr.constant(2)))));
                case "asin":
                case "arcsin":
                    return Expr.add(Expr.mul(u, Expr.func("asin", u)), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(u, Expr.constant(2)))));
                case "acos":
                case "arccos":
                    return Expr.sub(Expr.mul(u, Expr.func("acos", u)), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(u, Expr.constant(2)))));
                case "atan":
                case "arctan":
                    return Expr.sub(Expr.mul(u, Expr.func("atan", u)),
                            Expr.mul(Expr.constant(0.5), Expr.func("ln", Expr.add(Expr.constant(1), Expr.pow(u, Expr.constant(2))))));
                case "asinh":
                case "arcsinh":
                    return Expr.sub(Expr.mul(u, Expr.func("asinh", u)), Expr.func("sqrt", Expr.add(Expr.pow(u, Expr.constant(2)), Expr.constant(1))));
                case "acosh":
                case "arccosh":
                    return Expr.sub(Expr.mul(u, Expr.func("acosh", u)), Expr.func("sqrt", Expr.sub(Expr.pow(u, Expr.constant(2)), Expr.constant(1))));
                case "atanh":
                case "arctanh":
                    return Expr.add(Expr.mul(u, Expr.func("atanh", u)),
                            Expr.mul(Expr.constant(0.5), Expr.func("ln", Expr.sub(Expr.constant(1), Expr.pow(u, Expr.constant(2))))));
                case "abs":
                    return Expr.mul(Expr.constant(0.5), Expr.mul(u, Expr.func("abs", u)));
                default:
                    return null;
            }
        }

        // -------------------- LIATE priority (for integration by parts) --------------------
        int partsPriority(Expr e, String var) {
            if (e.kind == Expr.Kind.FUNC && e.children.length == 1) {
                switch (e.funcName) {
                    case "ln":
                    case "lg":
                        return 4;
                    case "asin":
                    case "arcsin":
                    case "acos":
                    case "arccos":
                    case "atan":
                    case "arctan":
                    case "asinh":
                    case "arcsinh":
                    case "acosh":
                    case "arccosh":
                    case "atanh":
                    case "arctanh":
                        return 3;
                    case "sin":
                    case "cos":
                    case "tan":
                    case "sec":
                    case "cosec":
                    case "csc":
                    case "cot":
                    case "sinh":
                    case "cosh":
                    case "tanh":
                    case "coth":
                    case "sech":
                    case "csch":
                    case "cosech":
                        return 1;
                    case "exp":
                        return 0;
                    default:
                        return -1;
                }
            }
            if (polyCoeffs(e, var, MAX_POLY_DEGREE) != null) {
                return 2;
            }
            return -1;
        }

        // -------------------- reciprocal-trig / auxiliary-angle helpers --------------------
        private static final class TrigTerm {

            final boolean isSin;
            final double coeff;
            final Expr arg;

            TrigTerm(boolean isSin, double coeff, Expr arg) {
                this.isSin = isSin;
                this.coeff = coeff;
                this.arg = arg;
            }
        }

        private static String reciprocalFuncName(String name) {
            switch (name) {
                case "sin":
                    return "cosec";
                case "cos":
                    return "sec";
                case "tan":
                    return "cot";
                case "cosec":
                case "csc":
                    return "sin";
                case "sec":
                    return "cos";
                case "cot":
                    return "tan";
                case "sinh":
                    return "csch";
                case "cosh":
                    return "sech";
                case "tanh":
                    return "coth";
                case "csch":
                case "cosech":
                    return "sinh";
                case "sech":
                    return "cosh";
                case "coth":
                    return "tanh";
                default:
                    return null;
            }
        }

        /**
         * Recognizes {@code k/f(u)} for trig/hyperbolic {@code f} and rewrites
         * it as {@code k*co-f(u)} (e.g. {@code 2/sin(x) ->
         * 2*csc(x)}, {@code 3/cos(4x) -> 3*sec(4x)}), then re-integrates via
         * the existing elementary table. {@code k} must not contain the
         * integration variable.
         */
        private Expr ruleReciprocalTrig(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.DIV) {
                return null;
            }
            Expr num = e.children[0], den = e.children[1];
            if (containsVar(num, var)) {
                return null;
            }
            if (den.kind != Expr.Kind.FUNC || den.children.length != 1) {
                return null;
            }
            String recip = reciprocalFuncName(den.funcName);
            if (recip == null) {
                return null;
            }
            Expr rewritten = Expr.mul(num, Expr.func(recip, den.children[0]));
            return integrate(rewritten, var, depth + 1);
        }

        /**
         * Extracts {@code (isSin, coeff, arg)} if {@code term} is
         * {@code +/- k*sin(u)} or {@code +/- k*cos(u)} for a literal constant
         * {@code k}; else null.
         */
        private TrigTerm extractSinOrCosTerm(Expr term, String var) {
            term = simplify(term);
            if (term.kind == Expr.Kind.FUNC && term.children.length == 1) {
                if ("sin".equals(term.funcName)) {
                    return new TrigTerm(true, 1.0, term.children[0]);
                }
                if ("cos".equals(term.funcName)) {
                    return new TrigTerm(false, 1.0, term.children[0]);
                }
                return null;
            }
            if (term.kind == Expr.Kind.NEG) {
                TrigTerm inner = extractSinOrCosTerm(term.children[0], var);
                return inner == null ? null : new TrigTerm(inner.isSin, -inner.coeff, inner.arg);
            }
            if (term.kind == Expr.Kind.MUL) {
                Expr a = term.children[0], b = term.children[1];
                if (a.kind == Expr.Kind.CONST) {
                    TrigTerm inner = extractSinOrCosTerm(b, var);
                    return inner == null ? null : new TrigTerm(inner.isSin, a.constVal * inner.coeff, inner.arg);
                }
                if (b.kind == Expr.Kind.CONST) {
                    TrigTerm inner = extractSinOrCosTerm(a, var);
                    return inner == null ? null : new TrigTerm(inner.isSin, b.constVal * inner.coeff, inner.arg);
                }
            }
            return null;
        }

        /**
         * {@code integral(k / (a*sin(u)+b*cos(u)))} via the auxiliary-angle
         * identity {@code a*sin(u)+b*cos(u) = R*sin(u+phi)},
         * {@code R=sqrt(a^2+b^2)}, {@code phi=atan2(b,a)} -- reduces to
         * {@code (k/R)*csc(u+phi)}, which the existing elementary table already
         * integrates.
         */
        private Expr ruleTrigLinearCombinationReciprocal(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.DIV) {
                return null;
            }
            Expr num = e.children[0], den = e.children[1];
            if (containsVar(num, var)) {
                return null;
            }
            if (den.kind != Expr.Kind.ADD && den.kind != Expr.Kind.SUB) {
                return null;
            }

            Expr left = den.children[0], right = den.children[1];
            if (den.kind == Expr.Kind.SUB) {
                right = Expr.neg(right);
            }
            TrigTerm t1 = extractSinOrCosTerm(left, var);
            TrigTerm t2 = extractSinOrCosTerm(right, var);
            if (t1 == null || t2 == null || t1.isSin == t2.isSin) {
                return null;
            }
            if (!deepEquals(simplify(t1.arg), simplify(t2.arg))) {
                return null;
            }

            double a = t1.isSin ? t1.coeff : t2.coeff;
            double b = t1.isSin ? t2.coeff : t1.coeff;
            double r = Math.sqrt(a * a + b * b);
            if (r < 1e-300) {
                return null;
            }
            double phi = Math.atan2(b, a);

            Expr u = t1.arg;
            double[] uAffine = affineCoeffs(u, var);
            if (uAffine == null || uAffine[0] == 0.0) {
                return null;
            }

            Expr v = Expr.add(u, Expr.constant(phi));
            Expr rewritten = Expr.mul(Expr.div(num, Expr.constant(r)), Expr.func("cosec", v));
            return integrate(rewritten, var, depth + 1);
        }

        // -------------------- default rule book --------------------
        private void registerDefaultRules() {
            rules.add(this::ruleNoVariable);
            rules.add(this::ruleVariableOrPowerOfVariable);
            rules.add(this::ruleSum);
            rules.add(this::ruleDifference);
            rules.add(this::ruleNegate);
            rules.add(this::ruleConstantMultiple);
            rules.add(this::rulePowerOfAffine);
            rules.add(this::ruleLinearArgumentTable);
            rules.add(this::ruleLogDerivative);
            rules.add(this::ruleRationalLinearOrQuadratic);
            rules.add(this::ruleReciprocalTrig);
            rules.add(this::ruleTrigLinearCombinationReciprocal);
            rules.add(this::ruleUSubstitution);
            rules.add(this::ruleIntegrationByParts);
        }

        private Expr ruleNoVariable(Expr e, String var, SymbolicEngine eng, int depth) {
            if (!containsVar(e, var)) {
                return Expr.mul(e, Expr.var(var));
            }
            return null;
        }

        private Expr ruleVariableOrPowerOfVariable(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind == Expr.Kind.VAR && e.varName.equals(var)) {
                return Expr.div(Expr.pow(e, Expr.constant(2)), Expr.constant(2));
            }
            return null;
        }

        private Expr ruleSum(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.ADD) {
                return null;
            }
            Expr ia = integrate(e.children[0], var, depth + 1);
            if (ia == null) {
                return null;
            }
            Expr ib = integrate(e.children[1], var, depth + 1);
            if (ib == null) {
                return null;
            }
            return Expr.add(ia, ib);
        }

        private Expr ruleDifference(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.SUB) {
                return null;
            }
            Expr ia = integrate(e.children[0], var, depth + 1);
            if (ia == null) {
                return null;
            }
            Expr ib = integrate(e.children[1], var, depth + 1);
            if (ib == null) {
                return null;
            }
            return Expr.sub(ia, ib);
        }

        private Expr ruleNegate(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.NEG) {
                return null;
            }
            Expr ia = integrate(e.children[0], var, depth + 1);
            return ia == null ? null : Expr.neg(ia);
        }

        private Expr ruleConstantMultiple(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.MUL) {
                return null;
            }
            Expr a = e.children[0], b = e.children[1];
            if (!containsVar(a, var)) {
                Expr ib = integrate(b, var, depth + 1);
                return ib == null ? null : Expr.mul(a, ib);
            }
            if (!containsVar(b, var)) {
                Expr ia = integrate(a, var, depth + 1);
                return ia == null ? null : Expr.mul(b, ia);
            }
            return null;
        }

        private Expr rulePowerOfAffine(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.POW || e.children[1].kind != Expr.Kind.CONST) {
                return null;
            }
            double[] ab = affineCoeffs(e.children[0], var);
            if (ab == null || ab[0] == 0.0) {
                return null;
            }
            double a = ab[0];
            double n = e.children[1].constVal;
            Expr base = e.children[0];
            if (n == -1.0) {
                return Expr.div(Expr.func("ln", Expr.func("abs", base)), Expr.constant(a));
            }
            return Expr.div(Expr.pow(base, Expr.constant(n + 1)), Expr.constant(a * (n + 1)));
        }

        private Expr ruleLinearArgumentTable(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.FUNC || e.children.length != 1) {
                return null;
            }
            double[] ab = affineCoeffs(e.children[0], var);
            if (ab == null || ab[0] == 0.0) {
                return null;
            }
            Expr H = elementaryAntiderivative(e.funcName, e.children[0]);
            if (H == null) {
                return null;
            }
            return Expr.div(H, Expr.constant(ab[0]));
        }

        private Expr ruleLogDerivative(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.DIV) {
                return null;
            }
            Expr a = e.children[0], b = e.children[1];
            if (!containsVar(b, var)) {
                return null;
            }
            Expr db;
            try {
                db = diff(b, var);
            } catch (RuntimeException ex) {
                return null;
            }
            Double k = constantRatio(a, db);
            if (k == null) {
                return null;
            }
            return Expr.mul(Expr.constant(k), Expr.func("ln", Expr.func("abs", b)));
        }

        private Expr ruleRationalLinearOrQuadratic(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.DIV) {
                return null;
            }
            double[] q = polyCoeffs(e.children[1], var, MAX_POLY_DEGREE);
            if (q == null) {
                return null;
            }
            int dq = degreeOf(q);
            if (dq != 1 && dq != 2) {
                return null;
            }
            double[] p = polyCoeffs(e.children[0], var, MAX_POLY_DEGREE);
            if (p == null) {
                return null;
            }
            int dp = degreeOf(p);

            double[] quotient, remainder;
            if (dp >= dq) {
                double[][] qr = polyDivide(p, q, dq);
                if (qr == null) {
                    return null;
                }
                quotient = qr[0];
                remainder = qr[1];
            } else {
                quotient = new double[]{0.0};
                remainder = p;
            }

            Expr quotientIntegral = null;
            if (!(quotient.length == 1 && quotient[0] == 0.0)) {
                quotientIntegral = integrate(polyToExpr(quotient, var), var, depth + 1);
                if (quotientIntegral == null) {
                    return null;
                }
            }

            Expr remainderPart;
            if (dq == 1) {
                double q1 = q[1];
                double r0 = remainder.length > 0 ? remainder[0] : 0.0;
                remainderPart = (r0 == 0.0) ? Expr.constant(0)
                        : Expr.mul(Expr.constant(r0 / q1), Expr.func("ln", Expr.func("abs", polyToExpr(q, var))));
            } else {
                double q2 = q[2], q1 = q[1], q0 = q[0];
                double r1 = remainder.length > 1 ? remainder[1] : 0.0;
                double r0 = remainder.length > 0 ? remainder[0] : 0.0;
                Expr Qexpr = polyToExpr(q, var);
                Expr logPart = (r1 == 0.0) ? Expr.constant(0)
                        : Expr.mul(Expr.constant(r1 / (2.0 * q2)), Expr.func("ln", Expr.func("abs", Qexpr)));
                double c2 = r0 - r1 * q1 / (2.0 * q2);
                Expr atanLnPart;
                if (c2 == 0.0) {
                    atanLnPart = Expr.constant(0);
                } else {
                    double h = q1 / (2.0 * q2);
                    double delta = q0 / q2 - h * h;
                    Expr xPlusH = Expr.sub(Expr.var(var), Expr.constant(-h));
                    if (delta > 1e-300) {
                        double s = Math.sqrt(delta);
                        atanLnPart = Expr.mul(Expr.constant(c2 / (q2 * s)), Expr.func("atan", Expr.div(xPlusH, Expr.constant(s))));
                    } else if (delta < -1e-300) {
                        double s = Math.sqrt(-delta);
                        atanLnPart = Expr.mul(Expr.constant(c2 / (2.0 * q2 * s)),
                                Expr.func("ln", Expr.func("abs", Expr.div(Expr.sub(xPlusH, Expr.constant(s)), Expr.add(xPlusH, Expr.constant(s))))));
                    } else {
                        atanLnPart = Expr.mul(Expr.constant(-c2 / q2), Expr.div(Expr.constant(1), xPlusH));
                    }
                }
                remainderPart = Expr.add(logPart, atanLnPart);
            }

            return quotientIntegral == null ? remainderPart : Expr.add(quotientIntegral, remainderPart);
        }

        private static int degreeOf(double[] coeffs) {
            for (int i = coeffs.length - 1; i >= 0; i--) {
                if (coeffs[i] != 0.0) {
                    return i;
                }
            }
            return 0;
        }

        private double[][] polyDivide(double[] p, double[] q, int dq) {
            double[] rem = p.clone();
            int dp = degreeOf(rem);
            int quotDeg = Math.max(0, dp - dq);
            double[] quot = new double[quotDeg + 1];
            double leadQ = q[dq];
            while (degreeOf(rem) >= dq && !(rem.length == 1 && rem[0] == 0.0)) {
                int dr = degreeOf(rem);
                if (dr < dq) {
                    break;
                }
                double coeff = rem[dr] / leadQ;
                int shift = dr - dq;
                if (shift > quotDeg) {
                    return null;
                }
                quot[shift] += coeff;
                for (int i = 0; i <= dq; i++) {
                    rem[i + shift] -= coeff * q[i];
                }
                if (dr == 0) {
                    break;
                }
            }
            return new double[][]{quot, rem};
        }

        private Expr ruleUSubstitution(Expr e, String var, SymbolicEngine eng, int depth) {
            if (depth > MAX_INTEGRATE_DEPTH - 2) {
                return null;
            }
            List<Expr> candidates = new ArrayList<>();
            collectCandidateInners(e, var, candidates);
            for (Expr g : candidates) {
                if (!containsVar(g, var)) {
                    continue;
                }
                if (g.kind == Expr.Kind.VAR) {
                    continue;
                }
                Expr dg;
                try {
                    dg = diff(g, var);
                } catch (RuntimeException ex) {
                    continue;
                }
                dg = simplify(dg);
                if (dg.kind == Expr.Kind.CONST && dg.constVal == 0.0) {
                    continue;
                }
                Expr ratio;
                try {
                    ratio = simplify(Expr.div(e, dg));
                } catch (RuntimeException ex) {
                    continue;
                }
                Expr substituted = replaceSubexpr(ratio, simplify(g), Expr.var(DUMMY_VAR));
                if (containsVar(substituted, var)) {
                    continue;
                }
                Expr recursiveResult = integrate(substituted, DUMMY_VAR, depth + 1);
                if (recursiveResult == null) {
                    continue;
                }
                return replaceSubexpr(recursiveResult, Expr.var(DUMMY_VAR), g);
            }
            return null;
        }

        private void collectCandidateInners(Expr e, String var, List<Expr> out) {
            if (out.size() > 24) {
                return;
            }
            if (e.kind == Expr.Kind.FUNC && e.children.length >= 1 && containsVar(e.children[0], var)) {
                addIfAbsent(out, e.children[0]);
            }
            if (e.kind == Expr.Kind.POW && containsVar(e.children[0], var)) {
                addIfAbsent(out, e.children[0]);
            }
            if (e.kind == Expr.Kind.MUL) {
                if (containsVar(e.children[0], var)) {
                    addIfAbsent(out, e.children[0]);
                }
                if (containsVar(e.children[1], var)) {
                    addIfAbsent(out, e.children[1]);
                }
            }
            for (Expr c : e.children) {
                collectCandidateInners(c, var, out);
            }
        }

        private void addIfAbsent(List<Expr> list, Expr candidate) {
            Expr s = simplify(candidate);
            for (Expr existing : list) {
                if (deepEquals(existing, s)) {
                    return;
                }
            }
            list.add(s);
        }

        private Expr ruleIntegrationByParts(Expr e, String var, SymbolicEngine eng, int depth) {
            if (depth > MAX_INTEGRATE_DEPTH - 2) {
                return null;
            }
            return tryByParts(e, e, var, depth, 0);
        }

        private Expr tryByParts(Expr originalIntegrand, Expr current, String var, int depth, int partsDepth) {
            if (partsDepth > MAX_BY_PARTS_DEPTH || current.kind != Expr.Kind.MUL) {
                return null;
            }
            Expr a = current.children[0], b = current.children[1];
            if (!containsVar(a, var) || !containsVar(b, var)) {
                return null;
            }

            int pa = partsPriority(a, var), pb = partsPriority(b, var);
            Expr[][] orderings = pa >= pb ? new Expr[][]{{a, b}, {b, a}} : new Expr[][]{{b, a}, {a, b}};

            for (Expr[] pair : orderings) {
                Expr u = pair[0], dv = pair[1];
                Expr du;
                try {
                    du = diff(u, var);
                } catch (RuntimeException ex) {
                    continue;
                }
                Expr V = integrate(dv, var, depth + 1);
                if (V == null) {
                    continue;
                }
                Expr boundary = simplify(Expr.mul(u, V));
                Expr remainder = simplify(Expr.mul(V, du));

                // Try the cheap, exact self-referential closure FIRST (matches the textbook
                // technique -- apply by-parts exactly twice, solve algebraically), rather than
                // the full unconstrained recursive integrate() on `remainder`, which for e.g.
                // exp(x)*sin(x) chases its own tail through further sin/cos/exp products and
                // burns the recursion-depth budget without ever reaching this closed form.
                if (partsDepth < MAX_BY_PARTS_DEPTH) {
                    Expr selfRef = trySelfReferential(originalIntegrand, remainder, boundary, var, depth, partsDepth + 1);
                    if (selfRef != null) {
                        return selfRef;
                    }
                }

                Expr direct = integrate(remainder, var, depth + 1);
                if (direct != null) {
                    return Expr.sub(boundary, direct);
                }
            }
            return null;
        }

        private Expr trySelfReferential(Expr originalIntegrand, Expr remainder, Expr boundaryFromFirstStep,
                String var, int depth, int partsDepth) {
            if (remainder.kind != Expr.Kind.MUL) {
                return null;
            }
            Expr a = remainder.children[0], b = remainder.children[1];
            if (!containsVar(a, var) || !containsVar(b, var)) {
                return null;
            }

            int pa = partsPriority(a, var), pb = partsPriority(b, var);
            Expr[][] orderings = pa >= pb ? new Expr[][]{{a, b}, {b, a}} : new Expr[][]{{b, a}, {a, b}};

            for (Expr[] pair : orderings) {
                Expr u2 = pair[0], dv2 = pair[1];
                Expr du2;
                try {
                    du2 = diff(u2, var);
                } catch (RuntimeException ex) {
                    continue;
                }
                Expr V2 = integrate(dv2, var, depth + 1);
                if (V2 == null) {
                    continue;
                }
                Expr boundary2 = simplify(Expr.mul(u2, V2));
                Expr remainder2 = simplify(Expr.mul(V2, du2));
                Double k = constantRatio(remainder2, originalIntegrand);
                if (k == null || Math.abs(1.0 - k) < 1e-12) {
                    continue;
                }
                Expr numerator = simplify(Expr.sub(boundaryFromFirstStep, boundary2));
                return Expr.div(numerator, Expr.constant(1.0 - k));
            }
            return null;
        }
    }

    // ------------------------------------------------------------------
    // SymbolicIntegrator: the public entry point
    // ------------------------------------------------------------------
    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {

        private double absTol = 1e-11;
        private double relTol = 1e-11;
        private int maxDepth = 40;
        private boolean strictSingularities = true;
        private boolean crossCheckEnabled = true;
        private boolean taylorRescueEnabled = true;
        private boolean interiorSingularityScanEnabled = true;
        private boolean symbolicEnabled = true;
        private NumericIntegrator explicitFallback;

        public Builder absoluteTolerance(double v) {
            this.absTol = v;
            return this;
        }

        public Builder relativeTolerance(double v) {
            this.relTol = v;
            return this;
        }

        public Builder maxDepth(int v) {
            this.maxDepth = v;
            return this;
        }

        public Builder strictSingularities(boolean v) {
            this.strictSingularities = v;
            return this;
        }

        public Builder crossCheckEnabled(boolean v) {
            this.crossCheckEnabled = v;
            return this;
        }

        public Builder taylorRescueEnabled(boolean v) {
            this.taylorRescueEnabled = v;
            return this;
        }

        public Builder interiorSingularityScanEnabled(boolean v) {
            this.interiorSingularityScanEnabled = v;
            return this;
        }

        public Builder symbolicEnabled(boolean v) {
            this.symbolicEnabled = v;
            return this;
        }

        /**
         * Overrides the default (auto-built {@link Integrator}) fallback with
         * any implementation.
         */
        public Builder fallback(NumericIntegrator fallback) {
            this.explicitFallback = fallback;
            return this;
        }

        public SymbolicIntegrator build(MathExpression me, String wrtVar) {
            return new SymbolicIntegrator(me, wrtVar, this);
        }

        public SymbolicIntegrator build(String expr, String wrtVar) {
            return build(new MathExpression(expr), wrtVar);
        }
    }

    private final MathExpression me;
    private final String wrtVar;
    private final Builder config;
    private final SymbolicEngine engine;
    private final Expr integrandExpr;

    private NumericIntegrator fallback;
    private boolean lastCallWasSymbolic;
    private Expr lastAntiderivative;
    private final String constructionParseFailureReason;
    private String lastSymbolicFailureReason;

    private SymbolicIntegrator(MathExpression me, String wrtVar, Builder config) {
        this.me = me;
        this.wrtVar = wrtVar;
        this.config = config;
        this.engine = new SymbolicEngine();
        this.fallback = config.explicitFallback;
        Token[] tokens = formatTokens(me.getCachedPostfix());

        Expr parsed = null;
        String parseFailure = null;
        if (config.symbolicEnabled) {
            try {
                parsed = engine.parseToExpr(tokens, wrtVar);
            } catch (RuntimeException parseIssue) {
                parseFailure = "symbolic parsing failed at construction (every call on this instance "
                        + "uses the numeric fallback): " + describeFully(parseIssue);
                LOG.warning(parseFailure);
            }
        } else {
            parseFailure = "symbolic path disabled via Builder.symbolicEnabled(false)";
        }
        this.integrandExpr = parsed;
        this.constructionParseFailureReason = parseFailure;
    }

    private static Token[] formatTokens(Token[] postfix) {
        Token[] rpn = new Token[postfix.length];
        for (int i = 0; i < postfix.length; i++) {
            Token t = postfix[i].clone();
            if (t.name != null && (t.name.endsWith("_rad") || t.name.endsWith("_grad") || t.name.endsWith("_deg"))) {
                t.name = t.name.substring(0, t.name.lastIndexOf("_"));
            }
            rpn[i] = t;
        }
        return rpn;
    }

    public SymbolicEngine getEngine() {
        return engine;
    }

    public double integrate(double a, double b) {
        lastCallWasSymbolic = false;
        lastAntiderivative = null;
        lastSymbolicFailureReason = null;

        if (integrandExpr == null) {
            lastSymbolicFailureReason = constructionParseFailureReason != null
                    ? constructionParseFailureReason : "symbolic path not attempted";
            return fallback().integrate(a, b);
        }

        try {
            Expr antideriv = engine.integrate(integrandExpr, wrtVar);
            if (antideriv == null) {
                lastSymbolicFailureReason = "no integration rule in the rule book matched this expression";
                return fallback().integrate(a, b);
            }
            Expr simplified = engine.simplify(antideriv);

            // Gate 1: derivative verification.
            if (!engine.verifyAntiderivative(integrandExpr, simplified, wrtVar, a, b)) {
                lastSymbolicFailureReason = "candidate antiderivative failed derivative verification "
                        + "(discarded for safety): " + simplified;
                return fallback().integrate(a, b);
            }

            // Gate 2: does the range look like it contains a singularity the 7 sample points
            // could have missed? Only checked if the configured fallback supports it.
            if (rangeLooksSuspicious(a, b)) {
                lastSymbolicFailureReason = "range flagged as possibly containing a singularity by the "
                        + "numeric fallback's own scan; not trusting the symbolic shortcut across it";
                return fallback().integrate(a, b);
            }

            double valAtB = engine.numericEval(simplified, wrtVar, b);
            double valAtA = engine.numericEval(simplified, wrtVar, a);
            double result = valAtB - valAtA;
            if (!Double.isFinite(result)) {
                lastSymbolicFailureReason = "symbolic antiderivative evaluated to a non-finite value at "
                        + "the bounds: " + simplified;
                return fallback().integrate(a, b);
            }

            // Gate 3: coarse independent numeric cross-check against the original integrand.
            Double coarse = coarseNumericCrossCheck(a, b);
            if (coarse != null) {
                double scale = Math.max(1.0, Math.abs(result));
                if (Math.abs(coarse - result) > 1e-3 * scale) {
                    lastSymbolicFailureReason = String.format(
                            "symbolic result (%.10g) disagreed with a coarse 32-panel numeric cross-check "
                            + "(%.10g) beyond tolerance; discarded for safety in favor of the fallback: %s",
                            result, coarse, simplified);
                    return fallback().integrate(a, b);
                }
            }
            // If the coarse cross-check itself couldn't be computed (a domain error somewhere in
            // [a,b], most likely a genuine pole), that is itself suspicious enough to distrust the
            // symbolic shortcut rather than assume the best.
            if (coarse == null) {
                lastSymbolicFailureReason = "coarse numeric cross-check could not be evaluated across "
                        + "[a,b] (likely a domain issue somewhere in range); discarded for safety: " + simplified;
                return fallback().integrate(a, b);
            }

            lastCallWasSymbolic = true;
            lastAntiderivative = simplified;
            return result;
        } catch (RuntimeException symbolicIssue) {
            lastSymbolicFailureReason = "symbolic integration threw " + describeFully(symbolicIssue);
            LOG.fine(lastSymbolicFailureReason);
            return fallback().integrate(a, b);
        }
    }

    private static String describeFully(Throwable t) {
        StringBuilder sb = new StringBuilder();
        sb.append(t.getClass().getName()).append(": ").append(t.getMessage());
        Throwable cause = t.getCause();
        while (cause != null) {
            sb.append(" [caused by ").append(cause.getClass().getName()).append(": ").append(cause.getMessage()).append(']');
            cause = cause.getCause();
        }
        return sb.toString();
    }

    /**
     * Gate 2: delegates to the fallback if it offers singularity-scanning;
     * otherwise not suspicious (gate skipped, gates 1 and 3 remain).
     */
    private boolean rangeLooksSuspicious(double a, double b) {
        NumericIntegrator fb = fallback();
        if (fb instanceof SingularityAwareIntegrator) {
            try {
                return ((SingularityAwareIntegrator) fb).rangeLooksSingular(a, b);
            } catch (RuntimeException ignore) {
                return true; // the check itself failing is its own bad sign -- err toward caution
            }
        }
        return false;
    }

    /**
     * Gate 3: cheap, direct composite Simpson on the original integrand. Null
     * if any sample throws or is non-finite.
     */
    private Double coarseNumericCrossCheck(double a, double b) {
        final int panels = 32; // even
        double lo = Math.min(a, b), hi = Math.max(a, b);
        double h = (hi - lo) / panels;
        try {
            double sum = engine.numericEval(integrandExpr, wrtVar, lo) + engine.numericEval(integrandExpr, wrtVar, hi);
            for (int i = 1; i < panels; i++) {
                double x = lo + i * h;
                double fx = engine.numericEval(integrandExpr, wrtVar, x);
                if (!Double.isFinite(fx)) {
                    return null;
                }
                sum += (i % 2 == 0 ? 2.0 : 4.0) * fx;
            }
            double result = sum * h / 3.0;
            if (!Double.isFinite(result)) {
                return null;
            }
            return (b < a) ? -result : result;
        } catch (EvalDomainException domainIssue) {
            return null;
        }
    }

    private NumericIntegrator fallback() {
        if (fallback == null) {
            fallback = Integrator.builder()
                    .absoluteTolerance(config.absTol)
                    .relativeTolerance(config.relTol)
                    .maxDepth(config.maxDepth)
                    .strictSingularities(config.strictSingularities)
                    .crossCheckEnabled(config.crossCheckEnabled)
                    .taylorRescueEnabled(config.taylorRescueEnabled)
                    .interiorSingularityScanEnabled(config.interiorSingularityScanEnabled)
                    .build(me, wrtVar);
        }
        return fallback;
    }

    /**
     * Why the most recent call used the numeric fallback, or null if it
     * succeeded symbolically. Always populated on a miss -- not gated behind
     * logging config.
     */
    public String getLastSymbolicFailureReason() {
        return lastSymbolicFailureReason;
    }

    public boolean wasLastResultSymbolic() {
        return lastCallWasSymbolic;
    }

    public String getLastAntiderivativeDescription() {
        return lastAntiderivative == null ? null : lastAntiderivative.toString();
    }

    // ------------------------------------------------------------------
    // Self-tests
    // ------------------------------------------------------------------
    private static int passCount = 0;
    private static int failCount = 0;

    private static void expect(String label, double actual, double expected, double tol,
            boolean expectSymbolic, boolean wasSymbolic, String failureReasonIfAny) {
        double err = Math.abs(actual - expected);
        boolean ok = err <= tol && expectSymbolic == wasSymbolic;
        System.out.printf("[%s] %-55s actual=%.12g expected=%.12g err=%.3e path=%s(expected %s)%n",
                ok ? "PASS" : "FAIL", label, actual, expected, err,
                wasSymbolic ? "SYMBOLIC" : "NUMERIC", expectSymbolic ? "SYMBOLIC" : "NUMERIC");
        if (expectSymbolic && !wasSymbolic && failureReasonIfAny != null) {
            System.out.println("       symbolic failure reason: " + failureReasonIfAny);
        }
        if (ok) {
            passCount++;
        } else {
            failCount++;
        }
    }

    private static SymbolicIntegrator make(String expr) {
        return builder().build(expr, "x");
    }

    public static void check1(String[] args) {
        SymbolicIntegrator poly = make("x^3+3*x^2-5*x-8");
        double vPoly = poly.integrate(-2, 3);
        expect("x^3+3x^2-5x-8 on [-2,3] (polynomial rule)", vPoly, -1.25, 1e-8, true, poly.wasLastResultSymbolic(), poly.getLastSymbolicFailureReason());

        SymbolicIntegrator sinI = make("sin(x)");
        double vSin = sinI.integrate(0, Math.PI);
        expect("sin(x) on [0,pi]", vSin, 2.0, 1e-8, true, sinI.wasLastResultSymbolic(), sinI.getLastSymbolicFailureReason());

        SymbolicIntegrator tanI = make("tan(x)");
        double vTan = tanI.integrate(0, 1);
        expect("tan(x) on [0,1]", vTan, -Math.log(Math.cos(1)), 1e-8, true, tanI.wasLastResultSymbolic(), tanI.getLastSymbolicFailureReason());

        SymbolicIntegrator usub = make("2*x*cos(x^2)");
        double vUsub = usub.integrate(0, 1.5);
        expect("2x*cos(x^2) on [0,1.5] (u-substitution, factor cancellation)", vUsub, Math.sin(1.5 * 1.5), 1e-7, true, usub.wasLastResultSymbolic(), usub.getLastSymbolicFailureReason());

        SymbolicIntegrator partsSelf = make("exp(x)*sin(x)");
        double vPartsSelf = partsSelf.integrate(0, 1);
        java.util.function.DoubleUnaryOperator F = x -> 0.5 * Math.exp(x) * (Math.sin(x) - Math.cos(x));
        double expPartsSelf = F.applyAsDouble(1) - F.applyAsDouble(0);
        expect("e^x*sin(x) on [0,1] (self-referential by parts)", vPartsSelf, expPartsSelf, 1e-6, true, partsSelf.wasLastResultSymbolic(), partsSelf.getLastSymbolicFailureReason());

        // --- New: reciprocal-trig cases ---
        SymbolicIntegrator recipSin = make("2/sin(x)");
        double vRecipSin = recipSin.integrate(1.0, 2.0);
        java.util.function.DoubleUnaryOperator FcscScaled = x -> -2.0 * Math.log(Math.abs(1.0 / Math.sin(x) + 1.0 / Math.tan(x)));
        double expRecipSin = FcscScaled.applyAsDouble(2.0) - FcscScaled.applyAsDouble(1.0);
        expect("2/sin(x) on [1,2] (reciprocal-trig rule)", vRecipSin, expRecipSin, 1e-6, true, recipSin.wasLastResultSymbolic(), recipSin.getLastSymbolicFailureReason());
        System.out.println("  F = " + recipSin.getLastAntiderivativeDescription());

        SymbolicIntegrator recipCos4x = make("3/cos(4*x)");
        double vRecipCos4x = recipCos4x.integrate(0.1, 0.3);
        java.util.function.DoubleUnaryOperator FsecScaled = x -> 0.75 * Math.log(Math.abs(1.0 / Math.cos(4 * x) + Math.tan(4 * x)));
        double expRecipCos4x = FsecScaled.applyAsDouble(0.3) - FsecScaled.applyAsDouble(0.1);
        expect("3/cos(4x) on [0.1,0.3] (reciprocal-trig rule, affine argument)", vRecipCos4x, expRecipCos4x, 1e-6, true, recipCos4x.wasLastResultSymbolic(), recipCos4x.getLastSymbolicFailureReason());
        System.out.println("  F = " + recipCos4x.getLastAntiderivativeDescription());

        // --- New: coth/sech/csch table entries ---
        SymbolicIntegrator cothI = make("1/tanh(x)");
        double vCoth = cothI.integrate(0.5, 1.5);
        double expCoth = Math.log(Math.abs(Math.sinh(1.5))) - Math.log(Math.abs(Math.sinh(0.5)));
        expect("1/tanh(x) on [0.5,1.5] (-> coth -> ln|sinh|)", vCoth, expCoth, 1e-6, true, cothI.wasLastResultSymbolic(), cothI.getLastSymbolicFailureReason());
        System.out.println("  F = " + cothI.getLastAntiderivativeDescription());

        SymbolicIntegrator sechI = make("1/cosh(x)");
        double vSech = sechI.integrate(0.0, 1.0);
        double expSech = Math.atan(Math.sinh(1.0)) - Math.atan(Math.sinh(0.0));
        expect("1/cosh(x) on [0,1] (-> sech -> atan(sinh))", vSech, expSech, 1e-6, true, sechI.wasLastResultSymbolic(), sechI.getLastSymbolicFailureReason());
        System.out.println("  F = " + sechI.getLastAntiderivativeDescription());

        SymbolicIntegrator cschI = make("1/sinh(x)");
        double vCsch = cschI.integrate(0.5, 1.5);
        double expCsch = Math.log(Math.abs(Math.tanh(1.5 / 2))) - Math.log(Math.abs(Math.tanh(0.5 / 2)));
        expect("1/sinh(x) on [0.5,1.5] (-> csch -> ln|tanh(x/2)|)", vCsch, expCsch, 1e-6, true, cschI.wasLastResultSymbolic(), cschI.getLastSymbolicFailureReason());
        System.out.println("  F = " + cschI.getLastAntiderivativeDescription());

        // --- New: the auxiliary-angle case ---
        SymbolicIntegrator sinPlusCos = make("1/(sin(x)+cos(x))");
        double vSinPlusCos = sinPlusCos.integrate(0.2, 0.8);
        java.util.function.DoubleUnaryOperator Faux = x -> (1.0 / Math.sqrt(2)) * Math.log(Math.abs(Math.tan(x / 2 + Math.PI / 8)));
        double expSinPlusCos = Faux.applyAsDouble(0.8) - Faux.applyAsDouble(0.2);
        expect("1/(sin(x)+cos(x)) on [0.2,0.8] (auxiliary-angle rule)", vSinPlusCos, expSinPlusCos, 1e-6, true, sinPlusCos.wasLastResultSymbolic(), sinPlusCos.getLastSymbolicFailureReason());
        System.out.println("  F = " + sinPlusCos.getLastAntiderivativeDescription());

        // --- Existing safety checks unchanged ---
        SymbolicIntegrator gaussian = make("exp(-x^2)");
        double vGaussian = gaussian.integrate(-3, 3);
        expect("exp(-x^2) on [-3,3] (NO elementary antiderivative -- must fall back)",
                vGaussian, 1.772414696519202, 1e-6, false, gaussian.wasLastResultSymbolic(), null);

        SymbolicIntegrator pole = make("1/(x-0.5)");
        try {
            double rp = pole.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x-0.5) through pole should have thrown via fallback, got " + rp);
            failCount++;
        } catch (Integrator.NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x-0.5) through pole correctly threw via fallback: " + expectedEx.getMessage());
            passCount++;
        }

        System.out.println();
        System.out.println(passCount + " passed, " + failCount + " failed");
        if (failCount > 0) {
            System.exit(1);
        }
    }

    public static void check2(String[] args) {
        SymbolicIntegrator poly = make("x^3+3*x^2-5*x-8");
        double vPoly = poly.integrate(-2, 3);
        expect("x^3+3x^2-5x-8 on [-2,3] (polynomial rule)", vPoly, -1.25, 1e-8, true, poly.wasLastResultSymbolic(), poly.getLastSymbolicFailureReason());

        SymbolicIntegrator sinI = make("sin(x)");
        double vSin = sinI.integrate(0, Math.PI);
        expect("sin(x) on [0,pi] (linear-argument table)", vSin, 2.0, 1e-8, true, sinI.wasLastResultSymbolic(), sinI.getLastSymbolicFailureReason());
        System.out.println("  F = " + sinI.getLastAntiderivativeDescription());

        SymbolicIntegrator invX = make("1/x");
        double vInvX = invX.integrate(1, 2);
        expect("1/x on [1,2] (power rule n=-1)", vInvX, Math.log(2), 1e-8, true, invX.wasLastResultSymbolic(), invX.getLastSymbolicFailureReason());

        SymbolicIntegrator tanI = make("tan(x)");
        double vTan = tanI.integrate(0, 1);
        expect("tan(x) on [0,1] (log-derivative rule)", vTan, -Math.log(Math.cos(1)), 1e-8, true, tanI.wasLastResultSymbolic(), tanI.getLastSymbolicFailureReason());
        System.out.println("  F = " + tanI.getLastAntiderivativeDescription());

        // u-substitution: integral(2x*cos(x^2)) = sin(x^2) -- FIXED: previously failed because the
        // constant "2" was buried inside a left-associated (2*x)*cos(x^2) and never got pulled out.
        SymbolicIntegrator usub = make("2*x*cos(x^2)");
        double vUsub = usub.integrate(0, 1.5);
        expect("2x*cos(x^2) on [0,1.5] (u-substitution, nested constant)", vUsub, Math.sin(1.5 * 1.5), 1e-7, true, usub.wasLastResultSymbolic(), usub.getLastSymbolicFailureReason());
        System.out.println("  F = " + usub.getLastAntiderivativeDescription());

        // A second, independent u-substitution case exercising the same nested-constant fix from a
        // different angle (constant multiplies the WHOLE product, not just one factor).
        SymbolicIntegrator usub2 = make("6*x^2*sin(x^3)");
        double vUsub2 = usub2.integrate(0, 1.2);
        double expUsub2 = -2.0 * (Math.cos(Math.pow(1.2, 3)) - Math.cos(0));
        expect("6x^2*sin(x^3) on [0,1.2] (u-substitution, degree-3 inner)", vUsub2, expUsub2, 1e-6, true, usub2.wasLastResultSymbolic(), usub2.getLastSymbolicFailureReason());
        System.out.println("  F = " + usub2.getLastAntiderivativeDescription());

        // Integration by parts (single application): integral(x*sin(x)) = sin(x) - x*cos(x)
        SymbolicIntegrator parts1 = make("x*sin(x)");
        double vParts1 = parts1.integrate(0, 2);
        double expParts1 = Math.sin(2) - 2 * Math.cos(2) - (Math.sin(0) - 0 * Math.cos(0));
        expect("x*sin(x) on [0,2] (integration by parts)", vParts1, expParts1, 1e-7, true, parts1.wasLastResultSymbolic(), parts1.getLastSymbolicFailureReason());
        System.out.println("  F = " + parts1.getLastAntiderivativeDescription());

        // Self-referential by parts: integral(e^x*sin(x)) -- FIXED: needed factor cancellation
        // (including NEG-as-(-1) unwrapping) to recognize -e^x*sin(x)/(e^x*sin(x)) == -1, and the
        // reordered self-reference check to actually reach that recognition.
        SymbolicIntegrator partsSelf = make("exp(x)*sin(x)");
        double vPartsSelf = partsSelf.integrate(0, 1);
        java.util.function.DoubleUnaryOperator F = x -> 0.5 * Math.exp(x) * (Math.sin(x) - Math.cos(x));
        double expPartsSelf = F.applyAsDouble(1) - F.applyAsDouble(0);
        expect("e^x*sin(x) on [0,1] (self-referential by parts)", vPartsSelf, expPartsSelf, 1e-6, true, partsSelf.wasLastResultSymbolic(), partsSelf.getLastSymbolicFailureReason());
        System.out.println("  F = " + partsSelf.getLastAntiderivativeDescription());

        // A second self-referential case with the roles of sin/cos swapped, for real evidence this
        // isn't a fix specific to one expression's exact shape.
        SymbolicIntegrator partsSelf2 = make("exp(x)*cos(x)");
        double vPartsSelf2 = partsSelf2.integrate(0, 1);
        java.util.function.DoubleUnaryOperator F2 = x -> 0.5 * Math.exp(x) * (Math.sin(x) + Math.cos(x));
        double expPartsSelf2 = F2.applyAsDouble(1) - F2.applyAsDouble(0);
        expect("e^x*cos(x) on [0,1] (self-referential by parts, swapped)", vPartsSelf2, expPartsSelf2, 1e-6, true, partsSelf2.wasLastResultSymbolic(), partsSelf2.getLastSymbolicFailureReason());
        System.out.println("  F = " + partsSelf2.getLastAntiderivativeDescription());

        // The previously-missing rule: f(x)/constant.
        SymbolicIntegrator constDenom = make("cos(x)/2");
        double vConstDenom = constDenom.integrate(0, 1);
        expect("cos(x)/2 on [0,1] (constant-denominator rule)", vConstDenom, Math.sin(1) / 2.0, 1e-9, true, constDenom.wasLastResultSymbolic(), constDenom.getLastSymbolicFailureReason());
        System.out.println("  F = " + constDenom.getLastAntiderivativeDescription());

        // Rational function, linear denominator, numerator degree >= denominator degree (needs division)
        SymbolicIntegrator ratLin = make("(x^2+1)/(x-3)");
        double vRatLin = ratLin.integrate(-1, 1);
        java.util.function.DoubleUnaryOperator FratLin = x -> x * x / 2.0 + 3 * x + 10 * Math.log(Math.abs(x - 3));
        double expRatLin = FratLin.applyAsDouble(1) - FratLin.applyAsDouble(-1);
        expect("(x^2+1)/(x-3) on [-1,1] (rational, linear denom + division)", vRatLin, expRatLin, 1e-7, true, ratLin.wasLastResultSymbolic(), ratLin.getLastSymbolicFailureReason());
        System.out.println("  F = " + ratLin.getLastAntiderivativeDescription());

        // Rational function, quadratic denominator, no real roots -> arctan form
        SymbolicIntegrator ratQuadAtan = make("1/(x^2+4)");
        double vRatQuadAtan = ratQuadAtan.integrate(0, 2);
        double expRatQuadAtan = (Math.atan(2.0 / 2.0) - Math.atan(0.0)) / 2.0;
        expect("1/(x^2+4) on [0,2] (rational, quadratic denom, arctan form)", vRatQuadAtan, expRatQuadAtan, 1e-8, true, ratQuadAtan.wasLastResultSymbolic(), ratQuadAtan.getLastSymbolicFailureReason());
        System.out.println("  F = " + ratQuadAtan.getLastAntiderivativeDescription());

        // Rational function, quadratic denominator, real roots -> ln form
        SymbolicIntegrator ratQuadLn = make("1/(x^2-9)");
        double vRatQuadLn = ratQuadLn.integrate(0, 2);
        double expRatQuadLn = (Math.log(Math.abs((2.0 - 3) / (2.0 + 3))) - Math.log(Math.abs((0.0 - 3) / (0.0 + 3)))) / 6.0;
        expect("1/(x^2-9) on [0,2] (rational, quadratic denom, ln form)", vRatQuadLn, expRatQuadLn, 1e-7, true, ratQuadLn.wasLastResultSymbolic(), ratQuadLn.getLastSymbolicFailureReason());
        System.out.println("  F = " + ratQuadLn.getLastAntiderivativeDescription());

        // --- Critical negative tests: genuinely non-elementary integrands MUST fail symbolically
        // and fall back to the numeric engine, not fabricate a wrong closed form.
        SymbolicIntegrator gaussian = make("exp(-x^2)");
        double vGaussian = gaussian.integrate(-3, 3);
        expect("exp(-x^2) on [-3,3] (NO elementary antiderivative -- must fall back)",
                vGaussian, 1.772414696519202, 1e-6, false, gaussian.wasLastResultSymbolic(), null);

        SymbolicIntegrator xx = make("x^x");
        double vXX = xx.integrate(1.1, 3);
        System.out.printf("[INFO ] x^x on [1.1,3] (generally non-elementary): got=%.16g  actual=%.20g, path=%s -- expected NUMERIC%n",
                vXX, 13.61975862562517599220026990647609183575,
                xx.wasLastResultSymbolic() ? "SYMBOLIC" : "NUMERIC");
        if (!xx.wasLastResultSymbolic()) {
            passCount++;
        } else {
            failCount++;
        } catch (Integrator.NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x-0.5) through pole correctly threw via fallback: " + expectedEx.getMessage());
            passCount++;
        }
         
        double vXX1 = xx.integrate(1.1, 15);
        System.out.printf("[INFO ] x^x on [1.1,15] (generally non-elementary): got=%.16g, actual=%.20g, path=%s -- expected NUMERIC%n",
                vXX1, 118685141706060739.36763292129980760724321639737101, 
                xx.wasLastResultSymbolic() ? "SYMBOLIC" : "NUMERIC");
        if (!xx.wasLastResultSymbolic()) {
            passCount++;
        } else {
            failCount++;
        }

        // Fallback still fully functional: an interior pole, handled entirely by the numeric engine's
        // own interior-singularity scan since the symbolic engine has no principal-value machinery.
        SymbolicIntegrator interiorPole = make("1/sin(x)");
        try {
            double rp = pole2.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x+3) through pole should have thrown via fallback, got " + rp);
            failCount++;
        } catch (Integrator.NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x+3) through pole correctly threw via fallback: " + expectedEx.getMessage());
            passCount++;
        }

        System.out.println();
        System.out.println(passCount + " passed, " + failCount + " failed");
        if (failCount > 0) {
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        check1(args);
        check2(args);
    }
    
}
