/** See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.numericalmethods.taylors.crx;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.logging.Logger;

/**
 * A real, rule-based symbolic integrator: parses the RPN token stream of an
 * expression into a small algebraic AST ({@link Expr}), attempts to find a
 * closed-form antiderivative via an ordered, expandable rule book ({@link
 * IntegrationRule}), and -- critically -- <b>verifies</b> any candidate
 * antiderivative (symbolic differentiation + numeric spot-checks) before
 * trusting it. If symbolic integration is not attempted, fails, or fails
 * verification, {@link SymbolicIntegrator#integrate} transparently falls
 * back to the fully-featured numeric {@link Integrator} built in prior
 * revisions of this codebase (Gauss-Kronrod, endpoint/interior singularity
 * handling, Taylor rescue, all of it) -- unchanged and undiminished.
 *
 * <h2>Honesty about scope</h2>
 * General symbolic integration is undecidable (this is literally what the
 * Risch algorithm exists to partially solve, and even it does not cover
 * everything). "Vast and expandable" here means something concrete: a real,
 * checkable rule book (see {@link SymbolicEngine} for the full list) plus a
 * mechanism ({@link SymbolicEngine#registerRule}) to add more -- not a claim
 * that every elementary function is covered. {@code exp(-x^2)} and general
 * {@code x^x} genuinely have no elementary closed-form antiderivative; this
 * class is expected to (and, per the self-tests, does) fail symbolically on
 * them and fall back cleanly, not fabricate an answer.
 *
 * <h2>Why verification is the load-bearing safety mechanism</h2>
 * Every rule in the book was hand-derived; some (the six inverse
 * trig/hyperbolic table entries, the by-parts self-reference algebra, the
 * complete-the-square quadratic-denominator cases) involve enough algebra
 * that a transcription slip is a real risk, and there is no compiler
 * available in this environment to catch one. Rather than trust hand-derived
 * calculus blindly, every candidate antiderivative {@code F} is symbolically
 * differentiated back to {@code F'} and spot-checked against the original
 * integrand at 7 points spread across the integration interval (excluding
 * the exact endpoints, to avoid boundary-singularity false failures) before
 * being used. A failing check is a hard, reliable veto -- it can never
 * produce a false <i>positive</i> (an actually-wrong result silently
 * delivered with confidence); it can only ever cause an unnecessary fall
 * back to the always-safe numeric engine. Passing is evidence, not
 * mathematical proof (finite sampling), which is exactly the same epistemic
 * standard applied to every other verification step built earlier in this
 * codebase (endpoint-subtraction re-probing, the GK/Taylor cross-check).
 *
 * <h2>Single-variable scope, deliberately</h2>
 * The parser rejects (bails to numeric) any {@code VARIABLE} token whose
 * name is not the integration variable. Every expression seen in this
 * codebase's own test suites is single-variable; multi-parameter symbolic
 * calculus is a materially larger feature (coefficients would need to carry
 * {@code Expr}, not {@code double}) that the numeric engine already handles
 * fully via its own variable-slot mechanism, so nothing is lost by
 * deferring it there.
 *
 * <h2>The rule book, concretely</h2>
 * Constant / power / sum / difference / constant-multiple / constant-
 * denominator rules; a table of ~25 elementary antiderivatives generalized
 * to any affine argument ({@code integral(h(ax+b)) = H(ax+b)/a}); the
 * logarithmic-derivative rule ({@code integral(f'/f) = ln|f|}, which
 * generically covers {@code tan}, {@code cot}, and any derivative-over-
 * function form); a genuine u-substitution search that recursively
 * re-invokes the whole engine on the transformed sub-problem; integration
 * by parts with LIATE-priority factor selection and the classic self-
 * referential "solve for I" technique (for {@code integral(e^x*sin(x))}-
 * shaped problems); and a complete-the-square, partial-fraction-free
 * solver for rational functions with linear or quadratic denominators
 * (numerator of any degree, via polynomial long division first). See
 * {@link SymbolicEngine}'s static initializer for the exact ordered rule
 * list.
 *
 * <h2>Revision: general versatility fixes, not point patches</h2>
 * Two originally-failing cases ({@code 2*x*cos(x^2)}, u-substitution;
 * {@code exp(x)*sin(x)}, self-referential by-parts) traced back to four
 * distinct, generally-applicable gaps rather than anything specific to
 * those two expressions:
 * <ol>
 *   <li><b>Nested-constant extraction.</b> {@code ruleConstantMultiple}
 *       only looked at a {@code MUL} node's two immediate children, so a
 *       constant buried one level deeper in a left-associated chain like
 *       {@code (2*x)*cos(x^2)} was never found. Now uses {@link
 *       #flattenMul}/{@link #rebuildMul} to gather every multiplicative
 *       factor regardless of nesting, partition constant from
 *       variable-dependent factors, and recurse on the product of the
 *       latter -- this also transitively strengthens u-substitution and
 *       by-parts, since both rely on the same recursive {@code integrate}
 *       call after a constant is pulled out.</li>
 *   <li><b>Division-side factor cancellation.</b> {@code simplify}'s
 *       {@code DIV} case previously only folded exact whole-operand
 *       matches; it had no notion of canceling a shared factor buried
 *       inside a product on each side (e.g. {@code x*cos(x^2) / (2*x)}
 *       never became {@code cos(x^2)/2}). This blocked u-substitution from
 *       ever recognizing that a candidate substitution actually eliminated
 *       the original variable. Fixed via the same flatten/rebuild
 *       machinery, applied to both operands of a division.</li>
 *   <li><b>Negation-aware cancellation.</b> {@code flattenMul} treats
 *       {@code -X} as the factor {@code -1} times {@code X}'s own factors,
 *       so e.g. {@code -sin(x)/sin(x)} now correctly cancels to the literal
 *       constant {@code -1} -- needed for the self-referential by-parts
 *       "solve for I" algebra to recognize its own closure at all.</li>
 *   <li><b>Self-reference search order.</b> {@code tryByParts} previously
 *       attempted the generic recursive {@code integrate} call on the
 *       by-parts remainder <i>before</i> the explicit self-referential
 *       check -- meaning that recursive call could reset context (a fresh
 *       {@code originalIntegrand}, a reset {@code partsDepth}) before the
 *       self-reference pattern was ever tried with the correct original
 *       problem in view. The self-referential check now runs first, with
 *       the generic recursive attempt as the fallback -- verified this
 *       reordering doesn't change behavior for an ordinary (non-self-
 *       referential) case like {@code x*sin(x)}, since {@code
 *       trySelfReferential} bails out immediately and cheaply whenever the
 *       by-parts remainder isn't itself a product of two var-dependent
 *       factors.</li>
 * </ol>
 * A fifth, smaller gap found in the same pass: there was no rule at all for
 * {@code f(x)/constant} (dividing by a plain number) -- {@code
 * ruleLogDerivative} and {@code ruleRationalLinearOrQuadratic} both
 * explicitly require the denominator to depend on the variable in a
 * specific way, so a bare constant denominator matched neither. Added
 * {@code ruleConstantDenominator} to close that gap. {@code deepEquals}
 * was also made commutative for {@code ADD}/{@code MUL} ({@code a*b}
 * and {@code b*a} now recognized as equal), which strengthens every
 * cancellation/dedup/matching operation built on top of it throughout the
 * engine, not just the two cases that motivated this revision.
 */
public final class SymbolicIntegrator {

    private static final Logger LOG = Logger.getLogger(SymbolicIntegrator.class.getName());

    // ------------------------------------------------------------------
    // Exceptions
    // ------------------------------------------------------------------

    /** Thrown internally whenever the symbolic engine meets a construct outside its scope. Always caught; never escapes {@link #integrate}. */
    public static final class UnsupportedSymbolicOperationException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        public UnsupportedSymbolicOperationException(String message) { super(message); }
    }

    /** Thrown by {@link SymbolicEngine#numericEval} on a domain error (log of a non-positive number, division by zero, etc). */
    public static final class EvalDomainException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        public EvalDomainException(String message) { super(message); }
    }

    // ------------------------------------------------------------------
    // Expr: minimal algebraic AST
    // ------------------------------------------------------------------

    public static final class Expr {
        public enum Kind { CONST, VAR, ADD, SUB, MUL, DIV, NEG, POW, FUNC }

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

        public static Expr constant(double v) { return new Expr(Kind.CONST, v, null, null, EMPTY); }
        public static Expr var(String name) { return new Expr(Kind.VAR, 0, name, null, EMPTY); }
        public static Expr add(Expr a, Expr b) { return new Expr(Kind.ADD, 0, null, null, new Expr[]{a, b}); }
        public static Expr sub(Expr a, Expr b) { return new Expr(Kind.SUB, 0, null, null, new Expr[]{a, b}); }
        public static Expr mul(Expr a, Expr b) { return new Expr(Kind.MUL, 0, null, null, new Expr[]{a, b}); }
        public static Expr div(Expr a, Expr b) { return new Expr(Kind.DIV, 0, null, null, new Expr[]{a, b}); }
        public static Expr neg(Expr a) { return new Expr(Kind.NEG, 0, null, null, new Expr[]{a}); }
        public static Expr pow(Expr a, Expr b) { return new Expr(Kind.POW, 0, null, null, new Expr[]{a, b}); }
        public static Expr func(String name, Expr... args) { return new Expr(Kind.FUNC, 0, null, name, args); }

        static final String UNSUPPORTED = "__unsupported__";

        @Override
        public String toString() {
            switch (kind) {
                case CONST: return trimNum(constVal);
                case VAR: return varName;
                case ADD: return "(" + children[0] + " + " + children[1] + ")";
                case SUB: return "(" + children[0] + " - " + children[1] + ")";
                case MUL: return "(" + children[0] + " * " + children[1] + ")";
                case DIV: return "(" + children[0] + " / " + children[1] + ")";
                case NEG: return "(-" + children[0] + ")";
                case POW: return "(" + children[0] + "^" + children[1] + ")";
                case FUNC: {
                    StringBuilder sb = new StringBuilder(funcName).append('(');
                    for (int i = 0; i < children.length; i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(children[i]);
                    }
                    return sb.append(')').toString();
                }
                default: return "?";
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
        /** Returns a (not-yet-simplified) candidate antiderivative, or null if this rule does not apply. */
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
                            String name = t.name;
                            if ("pow".equals(name)) {
                                throw new UnsupportedSymbolicOperationException("unary 'pow' unexpected");
                            }
                            stack.push(Expr.func(name, a));
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
                        // Structural no-ops in this postfix stream. AutoDiffNEvaluator's own
                        // evaluateRPN loop is an if/else-if chain with NO final else branch --
                        // meaning it silently skips any token kind it doesn't explicitly handle,
                        // including these. This parser must tolerate them identically.
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
                    case '+': stack.push(Expr.add(a, b)); break;
                    case '-': stack.push(Expr.sub(a, b)); break;
                    case '*': stack.push(Expr.mul(a, b)); break;
                    case '/': stack.push(Expr.div(a, b)); break;
                    case '^': stack.push(Expr.pow(a, b)); break;
                    default:
                        throw new UnsupportedSymbolicOperationException("unsupported binary operator '" + t.opChar + "'");
                }
            } else {
                Expr a = requirePop(stack);
                switch (t.opChar) {
                    case '-': stack.push(Expr.neg(a)); break;
                    case '²': stack.push(Expr.pow(a, Expr.constant(2))); break;
                    case '³': stack.push(Expr.pow(a, Expr.constant(3))); break;
                    case '√': stack.push(Expr.func("sqrt", a)); break;
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
            if (e.kind == Expr.Kind.VAR) return e.varName.equals(var);
            for (Expr c : e.children) {
                if (containsVar(c, var)) return true;
            }
            return false;
        }

        /** Commutative for ADD/MUL: a+b==b+a and a*b==b*a are both recognized. Strengthens every
         *  cancellation/dedup/matching operation built on this throughout the engine. */
        public boolean deepEquals(Expr a, Expr b) {
            if (a.kind != b.kind) return false;
            switch (a.kind) {
                case CONST: return a.constVal == b.constVal;
                case VAR: return a.varName.equals(b.varName);
                case FUNC: {
                    if (!a.funcName.equals(b.funcName) || a.children.length != b.children.length) return false;
                    for (int i = 0; i < a.children.length; i++) {
                        if (!deepEquals(a.children[i], b.children[i])) return false;
                    }
                    return true;
                }
                case ADD:
                case MUL:
                    return (deepEquals(a.children[0], b.children[0]) && deepEquals(a.children[1], b.children[1]))
                            || (deepEquals(a.children[0], b.children[1]) && deepEquals(a.children[1], b.children[0]));
                default: {
                    if (a.children.length != b.children.length) return false;
                    for (int i = 0; i < a.children.length; i++) {
                        if (!deepEquals(a.children[i], b.children[i])) return false;
                    }
                    return true;
                }
            }
        }

        /** Replaces every structurally-matching occurrence of {@code target} in {@code e} with {@code replacement}. */
        public Expr replaceSubexpr(Expr e, Expr target, Expr replacement) {
            if (deepEquals(e, target)) return replacement;
            if (e.children.length == 0) return e;
            Expr[] newChildren = new Expr[e.children.length];
            boolean changed = false;
            for (int i = 0; i < e.children.length; i++) {
                newChildren[i] = replaceSubexpr(e.children[i], target, replacement);
                if (newChildren[i] != e.children[i]) changed = true;
            }
            if (!changed) return e;
            switch (e.kind) {
                case ADD: return Expr.add(newChildren[0], newChildren[1]);
                case SUB: return Expr.sub(newChildren[0], newChildren[1]);
                case MUL: return Expr.mul(newChildren[0], newChildren[1]);
                case DIV: return Expr.div(newChildren[0], newChildren[1]);
                case NEG: return Expr.neg(newChildren[0]);
                case POW: return Expr.pow(newChildren[0], newChildren[1]);
                case FUNC: return Expr.func(e.funcName, newChildren);
                default: return e;
            }
        }

        /**
         * Flattens a MUL-chain (of any nesting/associativity) into its multiplicative factors.
         * {@code NEG(x)} is unwrapped as the factor {@code -1} followed by {@code x}'s own
         * factors -- this is what lets e.g. {@code -sin(x)/sin(x)} cancel down to the literal
         * constant {@code -1} rather than being left as an unrecognized {@code NEG(sin(x))/sin(x)}.
         */
        void flattenMul(Expr e, List<Expr> out) {
            if (e.kind == Expr.Kind.MUL) {
                flattenMul(e.children[0], out);
                flattenMul(e.children[1], out);
            } else if (e.kind == Expr.Kind.NEG) {
                out.add(Expr.constant(-1));
                flattenMul(e.children[0], out);
            } else {
                out.add(e);
            }
        }

        Expr rebuildMul(List<Expr> factors) {
            if (factors.isEmpty()) return Expr.constant(1);
            Expr result = factors.get(0);
            for (int i = 1; i < factors.size(); i++) {
                result = Expr.mul(result, factors.get(i));
            }
            return result;
        }

        // -------------------- simplification --------------------

        public Expr simplify(Expr e) {
            if (e.children.length == 0) return e;
            Expr[] sc = new Expr[e.children.length];
            for (int i = 0; i < sc.length; i++) sc[i] = simplify(e.children[i]);

            switch (e.kind) {
                case ADD: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) return Expr.constant(a.constVal + b.constVal);
                    if (a.kind == Expr.Kind.CONST && a.constVal == 0.0) return b;
                    if (b.kind == Expr.Kind.CONST && b.constVal == 0.0) return a;
                    return Expr.add(a, b);
                }
                case SUB: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) return Expr.constant(a.constVal - b.constVal);
                    if (b.kind == Expr.Kind.CONST && b.constVal == 0.0) return a;
                    if (deepEquals(a, b)) return Expr.constant(0);
                    if (a.kind == Expr.Kind.CONST && a.constVal == 0.0) return simplify(Expr.neg(b));
                    return Expr.sub(a, b);
                }
                case MUL: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) return Expr.constant(a.constVal * b.constVal);
                    if ((a.kind == Expr.Kind.CONST && a.constVal == 0.0) || (b.kind == Expr.Kind.CONST && b.constVal == 0.0)) return Expr.constant(0);
                    if (a.kind == Expr.Kind.CONST && a.constVal == 1.0) return b;
                    if (b.kind == Expr.Kind.CONST && b.constVal == 1.0) return a;
                    return Expr.mul(a, b);
                }
                case DIV: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST && b.constVal != 0.0) return Expr.constant(a.constVal / b.constVal);
                    if (b.kind == Expr.Kind.CONST && b.constVal == 1.0) return a;
                    if (a.kind == Expr.Kind.CONST && a.constVal == 0.0) return Expr.constant(0);
                    if (deepEquals(a, b)) return Expr.constant(1);

                    // Factor cancellation: flatten both sides into multiplicative factors and
                    // remove structurally-matching pairs (one from each side), regardless of
                    // nesting or which side a constant lives on. Fixes cases like
                    // x*cos(x^2)/(2*x) -> cos(x^2)/2, and -sin(x)/sin(x) -> -1 (via NEG unwrapping
                    // in flattenMul). Recurses (guaranteed to terminate: strictly fewer factors
                    // each time) so further folding after cancellation is picked up automatically.
                    List<Expr> numFactors = new ArrayList<>();
                    flattenMul(a, numFactors);
                    List<Expr> denFactors = new ArrayList<>();
                    flattenMul(b, denFactors);
                    boolean cancelled = false;
                    for (int i = denFactors.size() - 1; i >= 0; i--) {
                        Expr df = denFactors.get(i);
                        for (int j = 0; j < numFactors.size(); j++) {
                            if (deepEquals(numFactors.get(j), df)) {
                                numFactors.remove(j);
                                denFactors.remove(i);
                                cancelled = true;
                                break;
                            }
                        }
                    }
                    if (cancelled) {
                        return simplify(Expr.div(rebuildMul(numFactors), rebuildMul(denFactors)));
                    }
                    return Expr.div(a, b);
                }
                case NEG: {
                    Expr a = sc[0];
                    if (a.kind == Expr.Kind.CONST) return Expr.constant(-a.constVal);
                    if (a.kind == Expr.Kind.NEG) return a.children[0];
                    return Expr.neg(a);
                }
                case POW: {
                    Expr a = sc[0], b = sc[1];
                    if (a.kind == Expr.Kind.CONST && b.kind == Expr.Kind.CONST) {
                        double r = Math.pow(a.constVal, b.constVal);
                        if (Double.isFinite(r)) return Expr.constant(r);
                    }
                    if (b.kind == Expr.Kind.CONST && b.constVal == 1.0) return a;
                    if (b.kind == Expr.Kind.CONST && b.constVal == 0.0) return Expr.constant(1);
                    if (a.kind == Expr.Kind.POW && b.kind == Expr.Kind.CONST && a.children[1].kind == Expr.Kind.CONST) {
                        return simplify(Expr.pow(a.children[0], Expr.constant(a.children[1].constVal * b.constVal)));
                    }
                    return Expr.pow(a, b);
                }
                case FUNC: {
                    if (sc.length == 1 && sc[0].kind == Expr.Kind.CONST) {
                        Double folded = tryFoldUnary(e.funcName, sc[0].constVal);
                        if (folded != null) return Expr.constant(folded);
                    }
                    return Expr.func(e.funcName, sc);
                }
                default:
                    return e;
            }
        }

        private Double tryFoldUnary(String name, double x) {
            try {
                switch (name) {
                    case "sin": return Math.sin(x);
                    case "cos": return Math.cos(x);
                    case "exp": return Math.exp(x);
                    case "sqrt": return x >= 0 ? Math.sqrt(x) : null;
                    case "abs": return Math.abs(x);
                    case "ln": return x > 0 ? Math.log(x) : null;
                    default: return null;
                }
            } catch (RuntimeException ignore) {
                return null;
            }
        }

        // -------------------- differentiation --------------------

        public Expr diff(Expr e, String var) {
            switch (e.kind) {
                case CONST: return Expr.constant(0);
                case VAR: return Expr.constant(e.varName.equals(var) ? 1.0 : 0.0);
                case ADD: return simplify(Expr.add(diff(e.children[0], var), diff(e.children[1], var)));
                case SUB: return simplify(Expr.sub(diff(e.children[0], var), diff(e.children[1], var)));
                case NEG: return simplify(Expr.neg(diff(e.children[0], var)));
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
                    if (expoConst && baseConst) return Expr.constant(0);
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

        /** d/du f(u), expressed in terms of the given argument expr (chain rule multiplier applied by the caller). */
        private Expr funcDerivative(String name, Expr arg) {
            switch (name) {
                case "sin": return Expr.func("cos", arg);
                case "cos": return Expr.neg(Expr.func("sin", arg));
                case "tan": return Expr.div(Expr.constant(1), Expr.pow(Expr.func("cos", arg), Expr.constant(2)));
                case "sec": return Expr.div(Expr.func("sin", arg), Expr.pow(Expr.func("cos", arg), Expr.constant(2)));
                case "cosec": case "csc":
                    return Expr.neg(Expr.div(Expr.func("cos", arg), Expr.pow(Expr.func("sin", arg), Expr.constant(2))));
                case "cot": return Expr.neg(Expr.div(Expr.constant(1), Expr.pow(Expr.func("sin", arg), Expr.constant(2))));
                case "exp": return Expr.func("exp", arg);
                case "ln": return Expr.div(Expr.constant(1), arg);
                case "lg": return Expr.div(Expr.constant(1), Expr.mul(arg, Expr.constant(Math.log(10))));
                case "sqrt": return Expr.div(Expr.constant(1), Expr.mul(Expr.constant(2), Expr.func("sqrt", arg)));
                case "cbrt": return Expr.div(Expr.constant(1), Expr.mul(Expr.constant(3), Expr.pow(Expr.func("cbrt", arg), Expr.constant(2))));
                case "sinh": return Expr.func("cosh", arg);
                case "cosh": return Expr.func("sinh", arg);
                case "tanh": return Expr.div(Expr.constant(1), Expr.pow(Expr.func("cosh", arg), Expr.constant(2)));
                case "asin": case "arcsin":
                    return Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(arg, Expr.constant(2)))));
                case "acos": case "arccos":
                    return Expr.neg(Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(arg, Expr.constant(2))))));
                case "atan": case "arctan":
                    return Expr.div(Expr.constant(1), Expr.add(Expr.constant(1), Expr.pow(arg, Expr.constant(2))));
                case "asinh": case "arcsinh":
                    return Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.add(Expr.pow(arg, Expr.constant(2)), Expr.constant(1))));
                case "acosh": case "arccosh":
                    return Expr.div(Expr.constant(1), Expr.func("sqrt", Expr.sub(Expr.pow(arg, Expr.constant(2)), Expr.constant(1))));
                case "atanh": case "arctanh":
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
            if (depth > MAX_INTEGRATE_DEPTH) return null;
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
                case CONST: return e.constVal;
                case VAR: return e.varName.equals(var) ? x : 0.0;
                case ADD: return numericEval(e.children[0], var, x) + numericEval(e.children[1], var, x);
                case SUB: return numericEval(e.children[0], var, x) - numericEval(e.children[1], var, x);
                case MUL: return numericEval(e.children[0], var, x) * numericEval(e.children[1], var, x);
                case DIV: {
                    double denom = numericEval(e.children[1], var, x);
                    if (Math.abs(denom) < 1e-300) throw new EvalDomainException("division by zero");
                    return numericEval(e.children[0], var, x) / denom;
                }
                case NEG: return -numericEval(e.children[0], var, x);
                case POW: {
                    double base = numericEval(e.children[0], var, x);
                    double expo = numericEval(e.children[1], var, x);
                    double r = Math.pow(base, expo);
                    if (!Double.isFinite(r) && Double.isFinite(base) && Double.isFinite(expo)) {
                        throw new EvalDomainException("pow domain error: " + base + "^" + expo);
                    }
                    return r;
                }
                case FUNC: return evalFunc(e.funcName, e.children, var, x);
                default: throw new IllegalStateException("unreachable");
            }
        }

        private double evalFunc(String name, Expr[] children, String var, double x) {
            if (children.length == 2 && "atan2".equals(name)) {
                return Math.atan2(numericEval(children[0], var, x), numericEval(children[1], var, x));
            }
            double u = numericEval(children[0], var, x);
            switch (name) {
                case "sin": return Math.sin(u);
                case "cos": return Math.cos(u);
                case "tan": return Math.tan(u);
                case "sec": return 1.0 / Math.cos(u);
                case "cosec": case "csc": return 1.0 / Math.sin(u);
                case "cot": return 1.0 / Math.tan(u);
                case "exp": return Math.exp(u);
                case "ln": if (u <= 0) throw new EvalDomainException("ln domain"); return Math.log(u);
                case "lg": if (u <= 0) throw new EvalDomainException("lg domain"); return Math.log10(u);
                case "sqrt": if (u < 0) throw new EvalDomainException("sqrt domain"); return Math.sqrt(u);
                case "cbrt": return Math.cbrt(u);
                case "sinh": return Math.sinh(u);
                case "cosh": return Math.cosh(u);
                case "tanh": return Math.tanh(u);
                case "asin": case "arcsin":
                    if (Math.abs(u) > 1) throw new EvalDomainException("asin domain");
                    return Math.asin(u);
                case "acos": case "arccos":
                    if (Math.abs(u) > 1) throw new EvalDomainException("acos domain");
                    return Math.acos(u);
                case "atan": case "arctan": return Math.atan(u);
                case "asinh": case "arcsinh": return Math.log(u + Math.sqrt(u * u + 1));
                case "acosh": case "arccosh":
                    if (u < 1) throw new EvalDomainException("acosh domain");
                    return Math.log(u + Math.sqrt(u * u - 1));
                case "atanh": case "arctanh":
                    if (Math.abs(u) >= 1) throw new EvalDomainException("atanh domain");
                    return 0.5 * Math.log((1 + u) / (1 - u));
                case "abs": return Math.abs(u);
                default:
                    throw new EvalDomainException("unknown function for symbolic evaluation: " + name);
            }
        }

        // -------------------- verification --------------------

        /** Passing is evidence (finite sampling), not proof; failing is a hard, reliable veto. See class javadoc. */
        public boolean verifyAntiderivative(Expr integrand, Expr antiderivative, String var, double a, double b) {
            Expr derivative;
            try {
                derivative = simplify(diff(antiderivative, var));
            } catch (RuntimeException e) {
                return false;
            }
            double lo = Math.min(a, b), hi = Math.max(a, b);
            double width = hi - lo;
            if (width <= 0) return true;
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
                if (!Double.isFinite(expected) || !Double.isFinite(actual)) continue;
                double scale = Math.max(1.0, Math.max(Math.abs(expected), Math.abs(actual)));
                if (Math.abs(expected - actual) > 1e-6 * scale) {
                    return false;
                }
                checked++;
            }
            return checked > 0; // if every sample landed on a domain edge, we have no evidence -> don't trust it
        }

        // -------------------- affine / polynomial coefficient extraction --------------------

        /** Returns [a,b] such that e == a*var+b, or null if e is not affine in var. */
        double[] affineCoeffs(Expr e, String var) {
            switch (e.kind) {
                case CONST: return new double[]{0.0, e.constVal};
                case VAR: return new double[]{1.0, 0.0};
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
                    if (ca == null || cb == null) return null;
                    if (ca[0] == 0.0) return new double[]{ca[1] * cb[0], ca[1] * cb[1]};
                    if (cb[0] == 0.0) return new double[]{cb[1] * ca[0], cb[1] * ca[1]};
                    return null;
                }
                case DIV: {
                    double[] cb = affineCoeffs(e.children[1], var);
                    if (cb == null || cb[0] != 0.0 || cb[1] == 0.0) return null;
                    double[] ca = affineCoeffs(e.children[0], var);
                    return ca == null ? null : new double[]{ca[0] / cb[1], ca[1] / cb[1]};
                }
                case POW: {
                    if (e.children[1].kind == Expr.Kind.CONST) {
                        if (e.children[1].constVal == 0.0) return new double[]{0.0, 1.0};
                        if (e.children[1].constVal == 1.0) return affineCoeffs(e.children[0], var);
                    }
                    return null;
                }
                default:
                    return null;
            }
        }

        /** Coefficient array [c0,c1,...] (index = degree) if e is a polynomial in var of degree <= maxDegree; else null. */
        double[] polyCoeffs(Expr e, String var, int maxDegree) {
            switch (e.kind) {
                case CONST: return new double[]{e.constVal};
                case VAR: return new double[]{0.0, 1.0};
                case NEG: {
                    double[] c = polyCoeffs(e.children[0], var, maxDegree);
                    if (c == null) return null;
                    double[] r = c.clone();
                    for (int i = 0; i < r.length; i++) r[i] = -r[i];
                    return r;
                }
                case ADD: return addPoly(polyCoeffs(e.children[0], var, maxDegree), polyCoeffs(e.children[1], var, maxDegree), maxDegree);
                case SUB: {
                    double[] cb = polyCoeffs(e.children[1], var, maxDegree);
                    if (cb == null) return null;
                    double[] negb = cb.clone();
                    for (int i = 0; i < negb.length; i++) negb[i] = -negb[i];
                    return addPoly(polyCoeffs(e.children[0], var, maxDegree), negb, maxDegree);
                }
                case MUL: return mulPoly(polyCoeffs(e.children[0], var, maxDegree), polyCoeffs(e.children[1], var, maxDegree), maxDegree);
                case POW: {
                    if (e.children[1].kind == Expr.Kind.CONST) {
                        double nv = e.children[1].constVal;
                        if (nv == Math.floor(nv) && nv >= 0 && nv <= maxDegree) {
                            int n = (int) nv;
                            double[] base = polyCoeffs(e.children[0], var, maxDegree);
                            if (base == null) return null;
                            double[] result = new double[]{1.0};
                            for (int i = 0; i < n; i++) {
                                result = mulPoly(result, base, maxDegree);
                                if (result == null) return null;
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
            if (a == null || b == null) return null;
            int len = Math.max(a.length, b.length);
            if (len - 1 > maxDegree) return null;
            double[] r = new double[len];
            for (int i = 0; i < len; i++) r[i] = (i < a.length ? a[i] : 0) + (i < b.length ? b[i] : 0);
            return r;
        }

        private static double[] mulPoly(double[] a, double[] b, int maxDegree) {
            if (a == null || b == null) return null;
            int len = a.length + b.length - 1;
            if (len < 1) return new double[]{0.0};
            if (len - 1 > maxDegree) return null;
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
                if (coeffs[i] == 0.0) continue;
                Expr term = (i == 0) ? Expr.constant(coeffs[i])
                        : Expr.mul(Expr.constant(coeffs[i]), Expr.pow(Expr.var(var), Expr.constant(i)));
                result = Expr.add(result, term);
            }
            return simplify(result);
        }

        /** If a/b simplifies to a single literal constant, returns it; else null. Conservative by design -- see class javadoc. */
        Double constantRatio(Expr a, Expr b) {
            Expr ratio = simplify(Expr.div(a, b));
            return ratio.kind == Expr.Kind.CONST ? ratio.constVal : null;
        }

        // -------------------- elementary antiderivative table --------------------

        /** H(u) such that d/du H(u) = h(u), for the given function name; or null if h is not in the table. */
        Expr elementaryAntiderivative(String name, Expr u) {
            switch (name) {
                case "sin": return Expr.neg(Expr.func("cos", u));
                case "cos": return Expr.func("sin", u);
                case "tan": return Expr.neg(Expr.func("ln", Expr.func("abs", Expr.func("cos", u))));
                case "sec": return Expr.func("ln", Expr.func("abs", Expr.add(Expr.func("sec", u), Expr.func("tan", u))));
                case "cosec": case "csc":
                    return Expr.neg(Expr.func("ln", Expr.func("abs", Expr.add(Expr.func("cosec", u), Expr.func("cot", u)))));
                case "cot": return Expr.func("ln", Expr.func("abs", Expr.func("sin", u)));
                case "exp": return Expr.func("exp", u);
                case "ln": return Expr.sub(Expr.mul(u, Expr.func("ln", u)), u);
                case "lg": return Expr.div(Expr.sub(Expr.mul(u, Expr.func("ln", u)), u), Expr.constant(Math.log(10)));
                case "sqrt": return Expr.mul(Expr.constant(2.0 / 3.0), Expr.pow(u, Expr.constant(1.5)));
                case "cbrt": return Expr.mul(Expr.constant(0.75), Expr.pow(u, Expr.constant(4.0 / 3.0)));
                case "sinh": return Expr.func("cosh", u);
                case "cosh": return Expr.func("sinh", u);
                case "tanh": return Expr.func("ln", Expr.func("cosh", u));
                case "asin": case "arcsin":
                    return Expr.add(Expr.mul(u, Expr.func("asin", u)), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(u, Expr.constant(2)))));
                case "acos": case "arccos":
                    return Expr.sub(Expr.mul(u, Expr.func("acos", u)), Expr.func("sqrt", Expr.sub(Expr.constant(1), Expr.pow(u, Expr.constant(2)))));
                case "atan": case "arctan":
                    return Expr.sub(Expr.mul(u, Expr.func("atan", u)),
                            Expr.mul(Expr.constant(0.5), Expr.func("ln", Expr.add(Expr.constant(1), Expr.pow(u, Expr.constant(2))))));
                case "asinh": case "arcsinh":
                    return Expr.sub(Expr.mul(u, Expr.func("asinh", u)), Expr.func("sqrt", Expr.add(Expr.pow(u, Expr.constant(2)), Expr.constant(1))));
                case "acosh": case "arccosh":
                    return Expr.sub(Expr.mul(u, Expr.func("acosh", u)), Expr.func("sqrt", Expr.sub(Expr.pow(u, Expr.constant(2)), Expr.constant(1))));
                case "atanh": case "arctanh":
                    return Expr.add(Expr.mul(u, Expr.func("atanh", u)),
                            Expr.mul(Expr.constant(0.5), Expr.func("ln", Expr.sub(Expr.constant(1), Expr.pow(u, Expr.constant(2))))));
                case "abs": return Expr.mul(Expr.constant(0.5), Expr.mul(u, Expr.func("abs", u)));
                default: return null;
            }
        }

        // -------------------- LIATE priority (for integration by parts) --------------------

        int partsPriority(Expr e, String var) {
            if (e.kind == Expr.Kind.FUNC && e.children.length == 1) {
                switch (e.funcName) {
                    case "ln": case "lg": return 4;
                    case "asin": case "arcsin": case "acos": case "arccos": case "atan": case "arctan":
                    case "asinh": case "arcsinh": case "acosh": case "arccosh": case "atanh": case "arctanh":
                        return 3;
                    case "sin": case "cos": case "tan": case "sec": case "cosec": case "csc": case "cot":
                    case "sinh": case "cosh": case "tanh":
                        return 1;
                    case "exp": return 0;
                    default: return -1;
                }
            }
            if (polyCoeffs(e, var, MAX_POLY_DEGREE) != null) return 2;
            return -1;
        }

        // -------------------- default rule book --------------------

        private void registerDefaultRules() {
            rules.add(this::ruleNoVariable);
            rules.add(this::ruleVariableOrPowerOfVariable);
            rules.add(this::ruleSum);
            rules.add(this::ruleDifference);
            rules.add(this::ruleNegate);
            rules.add(this::ruleConstantMultiple);
            rules.add(this::ruleConstantDenominator);
            rules.add(this::rulePowerOfAffine);
            rules.add(this::ruleLinearArgumentTable);
            rules.add(this::ruleLogDerivative);
            rules.add(this::ruleRationalLinearOrQuadratic);
            rules.add(this::ruleUSubstitution);
            rules.add(this::ruleIntegrationByParts);
        }

        private Expr ruleNoVariable(Expr e, String var, SymbolicEngine eng, int depth) {
            if (!containsVar(e, var)) return Expr.mul(e, Expr.var(var));
            return null;
        }

        private Expr ruleVariableOrPowerOfVariable(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind == Expr.Kind.VAR && e.varName.equals(var)) {
                return Expr.div(Expr.pow(e, Expr.constant(2)), Expr.constant(2));
            }
            return null;
        }

        private Expr ruleSum(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.ADD) return null;
            Expr ia = integrate(e.children[0], var, depth + 1);
            if (ia == null) return null;
            Expr ib = integrate(e.children[1], var, depth + 1);
            if (ib == null) return null;
            return Expr.add(ia, ib);
        }

        private Expr ruleDifference(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.SUB) return null;
            Expr ia = integrate(e.children[0], var, depth + 1);
            if (ia == null) return null;
            Expr ib = integrate(e.children[1], var, depth + 1);
            if (ib == null) return null;
            return Expr.sub(ia, ib);
        }

        private Expr ruleNegate(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.NEG) return null;
            Expr ia = integrate(e.children[0], var, depth + 1);
            return ia == null ? null : Expr.neg(ia);
        }

        /**
         * Pulls every constant factor out of a MUL chain of ANY nesting/associativity (via {@link
         * #flattenMul}) -- not just the two immediate children of the top node -- and recurses on
         * the product of the variable-dependent factors. This is what lets {@code (2*x)*cos(x^2)}
         * (left-associated, constant buried one level deep) find its constant at all.
         */
        private Expr ruleConstantMultiple(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.MUL) return null;
            List<Expr> factors = new ArrayList<>();
            flattenMul(e, factors);
            List<Expr> constFactors = new ArrayList<>();
            List<Expr> varFactors = new ArrayList<>();
            for (Expr f : factors) {
                if (containsVar(f, var)) varFactors.add(f); else constFactors.add(f);
            }
            if (constFactors.isEmpty() || varFactors.isEmpty()) return null;
            Expr constProduct = rebuildMul(constFactors);
            Expr varProduct = rebuildMul(varFactors);
            Expr integratedVarPart = integrate(varProduct, var, depth + 1);
            if (integratedVarPart == null) return null;
            return Expr.mul(constProduct, integratedVarPart);
        }

        /**
         * integral(f(x)/k) = integral(f(x))/k for a var-free denominator k. A previously-missing
         * rule: neither {@link #ruleLogDerivative} nor {@link #ruleRationalLinearOrQuadratic}
         * matches a plain constant denominator (both require it to depend on the variable in a
         * specific way), so {@code cos(x)/2} had no rule at all before this.
         */
        private Expr ruleConstantDenominator(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.DIV) return null;
            Expr a = e.children[0], b = e.children[1];
            if (containsVar(b, var)) return null;
            Expr ia = integrate(a, var, depth + 1);
            return ia == null ? null : Expr.div(ia, b);
        }

        private Expr rulePowerOfAffine(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.POW || e.children[1].kind != Expr.Kind.CONST) return null;
            double[] ab = affineCoeffs(e.children[0], var);
            if (ab == null || ab[0] == 0.0) return null;
            double a = ab[0];
            double n = e.children[1].constVal;
            Expr base = e.children[0];
            if (n == -1.0) {
                return Expr.div(Expr.func("ln", Expr.func("abs", base)), Expr.constant(a));
            }
            return Expr.div(Expr.pow(base, Expr.constant(n + 1)), Expr.constant(a * (n + 1)));
        }

        private Expr ruleLinearArgumentTable(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.FUNC || e.children.length != 1) return null;
            double[] ab = affineCoeffs(e.children[0], var);
            if (ab == null || ab[0] == 0.0) return null;
            Expr H = elementaryAntiderivative(e.funcName, e.children[0]);
            if (H == null) return null;
            return Expr.div(H, Expr.constant(ab[0]));
        }

        private Expr ruleLogDerivative(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.DIV) return null;
            Expr a = e.children[0], b = e.children[1];
            if (!containsVar(b, var)) return null;
            Expr db;
            try {
                db = diff(b, var);
            } catch (RuntimeException ex) {
                return null;
            }
            Double k = constantRatio(a, db);
            if (k == null) return null;
            return Expr.mul(Expr.constant(k), Expr.func("ln", Expr.func("abs", b)));
        }

        private Expr ruleRationalLinearOrQuadratic(Expr e, String var, SymbolicEngine eng, int depth) {
            if (e.kind != Expr.Kind.DIV) return null;
            double[] q = polyCoeffs(e.children[1], var, MAX_POLY_DEGREE);
            if (q == null) return null;
            int dq = degreeOf(q);
            if (dq != 1 && dq != 2) return null;
            double[] p = polyCoeffs(e.children[0], var, MAX_POLY_DEGREE);
            if (p == null) return null;
            int dp = degreeOf(p);

            double[] quotient;
            double[] remainder;
            if (dp >= dq) {
                double[][] qr = polyDivide(p, q, dq);
                if (qr == null) return null;
                quotient = qr[0];
                remainder = qr[1];
            } else {
                quotient = new double[]{0.0};
                remainder = p;
            }

            Expr quotientIntegral = null;
            if (!(quotient.length == 1 && quotient[0] == 0.0)) {
                quotientIntegral = integrate(polyToExpr(quotient, var), var, depth + 1);
                if (quotientIntegral == null) return null;
            }

            Expr remainderPart;
            if (dq == 1) {
                double q1 = q[1], q0 = q[0];
                double r0 = remainder.length > 0 ? remainder[0] : 0.0;
                if (r0 == 0.0) {
                    remainderPart = Expr.constant(0);
                } else {
                    remainderPart = Expr.mul(Expr.constant(r0 / q1),
                            Expr.func("ln", Expr.func("abs", polyToExpr(q, var))));
                }
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
                if (coeffs[i] != 0.0) return i;
            }
            return 0;
        }

        /** Polynomial long division: p = quotient*q + remainder, remainder degree < dq. Returns {quotient, remainder} or null. */
        private double[][] polyDivide(double[] p, double[] q, int dq) {
            double[] rem = p.clone();
            int dp = degreeOf(rem);
            int quotDeg = Math.max(0, dp - dq);
            double[] quot = new double[quotDeg + 1];
            double leadQ = q[dq];
            while (degreeOf(rem) >= dq && !(rem.length == 1 && rem[0] == 0.0)) {
                int dr = degreeOf(rem);
                if (dr < dq) break;
                double coeff = rem[dr] / leadQ;
                int shift = dr - dq;
                if (shift > quotDeg) return null;
                quot[shift] += coeff;
                for (int i = 0; i <= dq; i++) {
                    rem[i + shift] -= coeff * q[i];
                }
                if (dr == 0) break;
            }
            return new double[][]{quot, rem};
        }

        private Expr ruleUSubstitution(Expr e, String var, SymbolicEngine eng, int depth) {
            if (depth > MAX_INTEGRATE_DEPTH - 2) return null;
            List<Expr> candidates = new ArrayList<>();
            collectCandidateInners(e, var, candidates);
            for (Expr g : candidates) {
                if (!containsVar(g, var)) continue;
                if (g.kind == Expr.Kind.VAR) continue;
                Expr dg;
                try {
                    dg = diff(g, var);
                } catch (RuntimeException ex) {
                    continue;
                }
                dg = simplify(dg);
                if (dg.kind == Expr.Kind.CONST && dg.constVal == 0.0) continue;
                Expr ratio;
                try {
                    ratio = simplify(Expr.div(e, dg));
                } catch (RuntimeException ex) {
                    continue;
                }
                Expr substituted = replaceSubexpr(ratio, simplify(g), Expr.var(DUMMY_VAR));
                if (containsVar(substituted, var)) continue;
                Expr recursiveResult = integrate(substituted, DUMMY_VAR, depth + 1);
                if (recursiveResult == null) continue;
                return replaceSubexpr(recursiveResult, Expr.var(DUMMY_VAR), g);
            }
            return null;
        }

        private void collectCandidateInners(Expr e, String var, List<Expr> out) {
            if (out.size() > 24) return;
            if (e.kind == Expr.Kind.FUNC && e.children.length >= 1 && containsVar(e.children[0], var)) {
                addIfAbsent(out, e.children[0]);
            }
            if (e.kind == Expr.Kind.POW && containsVar(e.children[0], var)) {
                addIfAbsent(out, e.children[0]);
            }
            if (e.kind == Expr.Kind.MUL) {
                if (containsVar(e.children[0], var)) addIfAbsent(out, e.children[0]);
                if (containsVar(e.children[1], var)) addIfAbsent(out, e.children[1]);
            }
            for (Expr c : e.children) {
                collectCandidateInners(c, var, out);
            }
        }

        private void addIfAbsent(List<Expr> list, Expr candidate) {
            Expr s = simplify(candidate);
            for (Expr existing : list) {
                if (deepEquals(existing, s)) return;
            }
            list.add(s);
        }

        private Expr ruleIntegrationByParts(Expr e, String var, SymbolicEngine eng, int depth) {
            if (depth > MAX_INTEGRATE_DEPTH - 2) return null;
            return tryByParts(e, e, var, depth, 0);
        }

        private Expr tryByParts(Expr originalIntegrand, Expr current, String var, int depth, int partsDepth) {
            if (partsDepth > MAX_BY_PARTS_DEPTH || current.kind != Expr.Kind.MUL) return null;
            Expr a = current.children[0], b = current.children[1];
            if (!containsVar(a, var) || !containsVar(b, var)) return null;

            int pa = partsPriority(a, var), pb = partsPriority(b, var);
            Expr[][] orderings = pa >= pb
                    ? new Expr[][]{{a, b}, {b, a}}
                    : new Expr[][]{{b, a}, {a, b}};

            for (Expr[] pair : orderings) {
                Expr u = pair[0], dv = pair[1];
                Expr du;
                try {
                    du = diff(u, var);
                } catch (RuntimeException ex) {
                    continue;
                }
                Expr V = integrate(dv, var, depth + 1);
                if (V == null) continue;
                Expr boundary = simplify(Expr.mul(u, V));
                Expr remainder = simplify(Expr.mul(V, du));

                // Self-referential check runs FIRST (fixed ordering): trying the generic recursive
                // integrate() call before this let it reset context (a fresh originalIntegrand, a
                // reset partsDepth) via its own internal by-parts attempt, so the self-reference
                // pattern for e.g. integral(e^x*sin(x)) was never reliably reached with the correct
                // original problem in view. trySelfReferential bails immediately and cheaply when
                // the remainder isn't itself a MUL of two var-dependent factors, so this reordering
                // costs nothing on an ordinary (non-self-referential) case like x*sin(x).
                if (partsDepth < MAX_BY_PARTS_DEPTH) {
                    Expr selfRef = trySelfReferential(originalIntegrand, remainder, boundary, var, depth, partsDepth + 1);
                    if (selfRef != null) return selfRef;
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
            if (remainder.kind != Expr.Kind.MUL) return null;
            Expr a = remainder.children[0], b = remainder.children[1];
            if (!containsVar(a, var) || !containsVar(b, var)) return null;

            int pa = partsPriority(a, var), pb = partsPriority(b, var);
            Expr[][] orderings = pa >= pb
                    ? new Expr[][]{{a, b}, {b, a}}
                    : new Expr[][]{{b, a}, {a, b}};

            for (Expr[] pair : orderings) {
                Expr u2 = pair[0], dv2 = pair[1];
                Expr du2;
                try {
                    du2 = diff(u2, var);
                } catch (RuntimeException ex) {
                    continue;
                }
                Expr V2 = integrate(dv2, var, depth + 1);
                if (V2 == null) continue;
                Expr boundary2 = simplify(Expr.mul(u2, V2));
                Expr remainder2 = simplify(Expr.mul(V2, du2));
                Double k = constantRatio(remainder2, originalIntegrand);
                if (k == null || Math.abs(1.0 - k) < 1e-12) continue;
                Expr numerator = simplify(Expr.sub(boundaryFromFirstStep, boundary2));
                return Expr.div(numerator, Expr.constant(1.0 - k));
            }
            return null;
        }
    }

    // ------------------------------------------------------------------
    // SymbolicIntegrator: the "neat client call" public entry point
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

        public Builder absoluteTolerance(double absTol) { this.absTol = absTol; return this; }
        public Builder relativeTolerance(double relTol) { this.relTol = relTol; return this; }
        public Builder maxDepth(int maxDepth) { this.maxDepth = maxDepth; return this; }
        public Builder strictSingularities(boolean strict) { this.strictSingularities = strict; return this; }
        public Builder crossCheckEnabled(boolean enabled) { this.crossCheckEnabled = enabled; return this; }
        public Builder taylorRescueEnabled(boolean enabled) { this.taylorRescueEnabled = enabled; return this; }
        public Builder interiorSingularityScanEnabled(boolean enabled) { this.interiorSingularityScanEnabled = enabled; return this; }

        /** Default true. When false, every call goes straight to the numeric engine (useful for A/B comparison or debugging). */
        public Builder symbolicEnabled(boolean enabled) { this.symbolicEnabled = enabled; return this; }

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

    private Integrator numericFallback;
    private boolean lastCallWasSymbolic;
    private Expr lastAntiderivative;
    private final String constructionParseFailureReason;
    private String lastSymbolicFailureReason;

    private SymbolicIntegrator(MathExpression me, String wrtVar, Builder config) {
        this.me = me;
        this.wrtVar = wrtVar;
        this.config = config;
        this.engine = new SymbolicEngine();

        Expr parsed = null;
        String parseFailure = null;
        if (config.symbolicEnabled) {
            try {
                Token[] tokens = formatTokens(me.getCachedPostfix());
                parsed = engine.parseToExpr(tokens, wrtVar);
            } catch (RuntimeException parseIssue) {
                parseFailure = "symbolic parsing failed at construction (every call on this instance will "
                        + "use the numeric engine): " + parseIssue;
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

    /** Exposes the engine so callers can register additional rules before integrating -- see class javadoc, "expandable". */
    public SymbolicEngine getEngine() {
        return engine;
    }

    public double integrate(double a, double b) {
        lastCallWasSymbolic = false;
        lastAntiderivative = null;
        lastSymbolicFailureReason = null;

        if (integrandExpr == null) {
            lastSymbolicFailureReason = constructionParseFailureReason != null
                    ? constructionParseFailureReason
                    : "symbolic path not attempted";
        } else {
            try {
                Expr antideriv = engine.integrate(integrandExpr, wrtVar);
                if (antideriv == null) {
                    lastSymbolicFailureReason = "no integration rule in the rule book matched this expression";
                } else {
                    Expr simplified = engine.simplify(antideriv);
                    if (!engine.verifyAntiderivative(integrandExpr, simplified, wrtVar, a, b)) {
                        lastSymbolicFailureReason = "candidate antiderivative failed numeric verification "
                                + "(discarded for safety rather than risk a wrong answer): " + simplified;
                    } else {
                        double valAtB = engine.numericEval(simplified, wrtVar, b);
                        double valAtA = engine.numericEval(simplified, wrtVar, a);
                        double result = valAtB - valAtA;
                        if (!Double.isFinite(result)) {
                            lastSymbolicFailureReason = "symbolic antiderivative evaluated to a non-finite "
                                    + "value at the bounds: " + simplified;
                        } else {
                            lastCallWasSymbolic = true;
                            lastAntiderivative = simplified;
                            return result;
                        }
                    }
                }
            } catch (RuntimeException symbolicIssue) {
                lastSymbolicFailureReason = "symbolic integration threw " + symbolicIssue.getClass().getSimpleName()
                        + ": " + symbolicIssue.getMessage();
                LOG.fine(lastSymbolicFailureReason);
            }
        }

        return numericFallback().integrate(a, b);
    }

    /**
     * Why the most recent call used the numeric fallback instead of a symbolic result -- null if
     * the most recent call succeeded symbolically. Always populated (not gated behind logging
     * configuration), specifically so a symbolic miss is a one-line answer to inspect rather than
     * something to reverse-engineer from output patterns.
     */
    public String getLastSymbolicFailureReason() {
        return lastSymbolicFailureReason;
    }

    private Integrator numericFallback() {
        if (numericFallback == null) {
            numericFallback = Integrator.builder()
                    .absoluteTolerance(config.absTol)
                    .relativeTolerance(config.relTol)
                    .maxDepth(config.maxDepth)
                    .strictSingularities(config.strictSingularities)
                    .crossCheckEnabled(config.crossCheckEnabled)
                    .taylorRescueEnabled(config.taylorRescueEnabled)
                    .interiorSingularityScanEnabled(config.interiorSingularityScanEnabled)
                    .build(me, wrtVar);
        }
        return numericFallback;
    }

    public boolean wasLastResultSymbolic() {
        return lastCallWasSymbolic;
    }

    /** Human-readable antiderivative from the most recent call, or null if that call used the numeric fallback. */
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
        boolean numOk = err <= tol;
        boolean pathOk = (expectSymbolic == wasSymbolic);
        boolean ok = numOk && pathOk;
        System.out.printf("[%s] %-55s actual=%.12g expected=%.12g err=%.3e path=%s(expected %s)%n",
                ok ? "PASS" : "FAIL", label, actual, expected, err,
                wasSymbolic ? "SYMBOLIC" : "NUMERIC", expectSymbolic ? "SYMBOLIC" : "NUMERIC");
        if (expectSymbolic && !wasSymbolic && failureReasonIfAny != null) {
            System.out.println("       symbolic failure reason: " + failureReasonIfAny);
        }
        if (ok) passCount++; else failCount++;
    }

    private static SymbolicIntegrator make(String expr) {
        return builder().build(expr, "x");
    }

    public static void main(String[] args) {
        SymbolicIntegrator poly = make("x^3+3*x^2-5*x-8");
        double vPoly = poly.integrate(-2, 3);
        expect("x^3+3x^2-5x-8 on [-2,3] (polynomial rule)", vPoly, -1.25, 1e-8, true, poly.wasLastResultSymbolic(), poly.getLastSymbolicFailureReason());
        System.out.println("  F = " + poly.getLastAntiderivativeDescription());

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
        System.out.printf("[INFO ] x^x on [1.1,3] (generally non-elementary): got=%.10g path=%s -- expected NUMERIC%n",
                vXX, xx.wasLastResultSymbolic() ? "SYMBOLIC" : "NUMERIC");
        if (!xx.wasLastResultSymbolic()) passCount++; else failCount++;

        // Fallback still fully functional: an interior pole, handled entirely by the numeric engine's
        // own interior-singularity scan since the symbolic engine has no principal-value machinery.
        SymbolicIntegrator interiorPole = make("1/sin(x)");
        try {
            interiorPole.integrate(1, 4);
            System.out.println("[FAIL] 1/sin(x) on [1,4] (crosses pole at pi) should have thrown via numeric fallback");
            failCount++;
        } catch (Integrator.NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/sin(x) on [1,4] correctly threw via numeric fallback: " + expectedEx.getMessage());
            passCount++;
        }

        System.out.println();
        System.out.println(passCount + " passed, " + failCount + " failed");
        if (failCount > 0) {
            System.exit(1);
        }
    }
} 
