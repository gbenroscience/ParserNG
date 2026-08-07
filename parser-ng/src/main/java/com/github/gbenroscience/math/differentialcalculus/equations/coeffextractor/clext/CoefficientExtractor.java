/*
 * Copyright 2026 GBEMIRO.
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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;
 
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author GBEMIRO
 * Extracts per-derivative-order coefficients from an equation of the general
 * form ParserNG's diffeqn/diffeqnPath/diffeqnHO/diffeqnHOPath accept as
 * their first argument:
 *
 * <pre>
 *   A1(t)*y[n] + A2(t)*f1(y[n-1]) + A3(t)*f2(y[n-2]) + ... + Ak(t)*fn(y[0]) - B(t)
 * </pre>
 *
 * where each A_i(t) is a coefficient (function of t, or a constant), each
 * f_i is at most one wrapping function applied directly to that term's
 * state variable (identity — no wrap at all — is allowed and common; see
 * -3*y[1] and 3*x*y[0] in the worked example below), B(t) is an optional
 * forcing term with no state reference at all, and the equation is written
 * with an implicit "= 0" — there is no separate right-hand side argument to
 * this class, unlike LinearFormExtractor/TopDerivativeExtractor which
 * pre-date this looser form. The one required invariant, exactly as
 * specified: <b>the top-order term (A1(t)*y[n]) must always be strictly
 * linear — no wrapping function is allowed there</b>, even though every
 * other order may have one.
 *
 * <h2>Three ways in</h2>
 * <ul>
 *   <li>{@link #extract(String)} — a raw expression string, lexed here via
 *       {@link Lexer} then parsed via {@link ExprParser}.</li>
 *   <li>{@link #extract(List)} — an already-scanned flat token list, exactly
 *       the form shown in ParserNG's own scanner output (one array entry per
 *       atomic lexeme — parens, brackets, operators, digits, letters).</li>
 *   <li>{@link #extract(ExprNode)} — an already-parsed expression tree, for
 *       callers that have their own parser and just want the coefficient
 *       extraction logic.</li>
 * </ul>
 * The first two both accept either the bare equation on its own, or the
 * <em>entire</em> scanned/raw {@code diffeqn(...)} call (equation plus t0,
 * y0, tEnd, method, etc.) — {@link ArgumentIsolator} detects which case it
 * is and, for the latter, isolates just the first argument before parsing.
 * This is what makes it safe to hand this class the literal output of
 * ParserNG's own scanner without pre-processing it yourself.
 *
 * <h2>What counts as valid, precisely</h2>
 * The equation is split into additive terms (top-level + and -, exactly as
 * in LinearFormExtractor). Each term must reference at most one derivative
 * order:
 * <ul>
 *   <li><b>Zero state references</b> — the term is forcing-term (B(t))
 *       material; multiple such terms across the equation are summed.</li>
 *   <li><b>Exactly one state reference, y[k]</b> — accepted if either
 *       (a) y[k] is a plain multiplicative factor of the term (the identity
 *       case — no function wraps it), or (b) y[k]'s immediate parent in the
 *       tree is a single-argument function call (e.g. sin(y[k])), and that
 *       whole function-call node is itself a plain multiplicative factor of
 *       the term. Anything deeper — two nested functions, y[k] as an
 *       argument among several in a multi-arg call, y[k]^2, y[k] as a
 *       divisor — is rejected with a specific reason, not silently
 *       misclassified.</li>
 *   <li><b>More than one state reference</b> (whether the same order twice,
 *       e.g. y[2]*y[2], or two different orders, e.g. y[2]*y[1]) is
 *       rejected: this form has no provision for coupling between orders.</li>
 * </ul>
 * If the same order k is found in more than one distinct additive term
 * (e.g. "sin(y[2]) + cos(y[2])"), that is also rejected — this form
 * specifies exactly one A_k(t)*f_k(y[n-k]) per order, so which of two
 * competing terms should "win" is genuinely ambiguous rather than something
 * to guess at.
 *
 * Any order between 0 and the top order that never appears in the equation
 * at all is filled in with coefficient 0 (identity wrap) — the array and map
 * outputs are always dense across 0..topOrder, never sparse.
 */
public final class CoefficientExtractor {

    private CoefficientExtractor() {
    }

    // ------------------------------------------------------------------
    // Result shape
    // ------------------------------------------------------------------

    /** One derivative order's contribution, as it was actually written. */
    public static final class DerivativeTerm {
        public final int order;
        /** null if this order was unwrapped (identity) or was absent entirely. */
        public final String functionName;
        /** A_k(t), sign already folded in — coefficient * stateFactor reconstructs the original term exactly. */
        public final ExprNode coefficient;
        /** The bare y[order] leaf, or the whole f(y[order]) call if wrapped. */
        public final ExprNode stateFactor;
        /** True if this order never actually appeared in the equation (coefficient is a synthetic 0). */
        public final boolean absent;

        DerivativeTerm(int order, String functionName, ExprNode coefficient, ExprNode stateFactor, boolean absent) {
            this.order = order;
            this.functionName = functionName;
            this.coefficient = coefficient;
            this.stateFactor = stateFactor;
            this.absent = absent;
        }

        void appendJson(StringBuilder sb) {
            sb.append("{\"order\":").append(order);
            sb.append(",\"functionName\":");
            ExprNode.appendJsonString(sb, functionName);
            sb.append(",\"coefficient\":");
            if (coefficient == null) {
                sb.append("null");
            } else {
                coefficient.appendJson(sb);
            }
            sb.append(",\"stateFactor\":");
            if (stateFactor == null) {
                sb.append("null");
            } else {
                stateFactor.appendJson(sb);
            }
            sb.append(",\"absent\":").append(absent);
            sb.append('}');
        }

        /** Renders this term as JSON: order, functionName, coefficient, stateFactor, absent. */
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            appendJson(sb);
            return sb.toString();
        }
    }

    public static final class Result {
        public final int topOrder;

        /** Ordered highest order first: terms.get(0) is order topOrder, terms.get(topOrder) is order 0. */
        public final List<DerivativeTerm> terms;

        /** B(t), or null if the equation had no state-free term at all. */
        public final ExprNode forcingOrNull;

        /** Keyed by derivative order k (0..topOrder); every key present. */
        public final Map<Integer, ExprNode> coefficientsByOrder;

        /** Parallel to derivativeTerms below, highest order first, length topOrder+1. */
        public final ExprNode[] coefficients;

        /**
         * Parallel to coefficients above. coefficients[i] * derivativeTerms[i]
         * reconstructs the i-th term of the equation exactly (sign included
         * in coefficients[i]) — this is the array pairing requested: same
         * order, corresponding entries multiply back to the original term.
         */
        public final ExprNode[] derivativeTerms;

        /** Parallel to the two arrays above; null entry means that order was unwrapped (identity). */
        public final String[] wrappingFunctionNames;

        Result(int topOrder, List<DerivativeTerm> terms, ExprNode forcingOrNull) {
            this.topOrder = topOrder;
            this.terms = terms;
            this.forcingOrNull = forcingOrNull;

            this.coefficientsByOrder = new LinkedHashMap<>();
            this.coefficients = new ExprNode[terms.size()];
            this.derivativeTerms = new ExprNode[terms.size()];
            this.wrappingFunctionNames = new String[terms.size()];
            for (int i = 0; i < terms.size(); i++) {
                DerivativeTerm dt = terms.get(i);
                coefficientsByOrder.put(dt.order, dt.coefficient);
                coefficients[i] = dt.coefficient;
                derivativeTerms[i] = dt.stateFactor;
                wrappingFunctionNames[i] = dt.functionName;
            }
        }

        /**
         * Renders this result as JSON: {@code topOrder}, the full
         * {@code terms} list (highest order first, each as its own JSON
         * object via {@link DerivativeTerm#toString()}), and
         * {@code forcingOrNull}. The three parallel arrays and the
         * order-keyed map are intentionally left out here since they're
         * fully derivable from {@code terms} and would just duplicate it.
         */
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"topOrder\":").append(topOrder);
            sb.append(",\"terms\":[");
            for (int i = 0; i < terms.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                terms.get(i).appendJson(sb);
            }
            sb.append(']');
            sb.append(",\"forcingOrNull\":");
            if (forcingOrNull == null) {
                sb.append("null");
            } else {
                forcingOrNull.appendJson(sb);
            }
            sb.append('}');
            return sb.toString();
        }
    }

    // ------------------------------------------------------------------
    // Entry points
    // ------------------------------------------------------------------

    public static Result extract(String rawExpressionOrFullCall) {
        return extract(Lexer.tokenize(rawExpressionOrFullCall));
    }

    public static Result extract(List<String> scannedTokensOrFullCall) {
        List<String> equationTokens = ArgumentIsolator.isolateFirstArgument(scannedTokensOrFullCall);
        ExprNode expression = ExprParser.parse(equationTokens);
        return extract(expression);
    }

    /**
     * @param expression the whole equation, implicitly "expression = 0" —
     *                    matching the form's own convention of folding any
     *                    right-hand side in as a subtracted term
     */
    public static Result extract(ExprNode expression) {
        List<ExprAlgebra.SignedTerm> terms = ExprAlgebra.collectAdditiveTerms(expression);

        Map<Integer, PerOrderMatch> matchesByOrder = new LinkedHashMap<>();
        List<ExprNode> forcingParts = new ArrayList<>();
        int topOrder = -1;

        for (ExprAlgebra.SignedTerm term : terms) {
            List<ExprNode> leaves = new ArrayList<>();
            ExprAlgebra.findStateLeaves(term.node, leaves);

            if (leaves.isEmpty()) {
                forcingParts.add(ExprAlgebra.applySign(term.node, term.sign));
                continue;
            }
            if (leaves.size() > 1) {
                throw new IllegalArgumentException(
                        "Term " + ExprAlgebra.describe(term.node) + " references the state more than once — "
                        + "this form allows at most one derivative order per additive term.");
            }

            ExprNode leaf = leaves.get(0);
            PerOrderMatch match = classifyTerm(term.node, term.sign, leaf);

            if (matchesByOrder.containsKey(leaf.stateIndex)) {
                throw new IllegalArgumentException(
                        "y[" + leaf.stateIndex + "] appears in more than one additive term — this form "
                        + "specifies exactly one coefficient (and at most one wrapping function) per order.");
            }
            matchesByOrder.put(leaf.stateIndex, match);
            topOrder = Math.max(topOrder, leaf.stateIndex);
        }

        if (topOrder < 0) {
            throw new IllegalArgumentException("Equation has no state-variable term at all — nothing to solve for.");
        }

        PerOrderMatch topMatch = matchesByOrder.get(topOrder);
        if (topMatch != null && topMatch.functionName != null) {
            throw new IllegalArgumentException(
                    "The top-order term y[" + topOrder + "] is wrapped in " + topMatch.functionName
                    + "(...) — the top derivative must always be strictly linear (A1(t)*y[" + topOrder
                    + "], no wrapping function), per this form's own rule.");
        }

        List<DerivativeTerm> ordered = new ArrayList<>();
        for (int k = topOrder; k >= 0; k--) {
            PerOrderMatch m = matchesByOrder.get(k);
            if (m == null) {
                ordered.add(new DerivativeTerm(k, null, ExprNode.number(0.0), ExprNode.stateVariable("y", k), true));
            } else {
                ordered.add(new DerivativeTerm(k, m.functionName, m.coefficient, m.stateFactor, false));
            }
        }

        ExprNode forcing = forcingParts.isEmpty() ? null : ExprAlgebra.applySign(ExprAlgebra.sumAll(forcingParts), -1);

        return new Result(topOrder, ordered, forcing);
    }

    // ------------------------------------------------------------------
    // Per-term classification: identity or exactly-one-function-wrap
    // ------------------------------------------------------------------

    private static final class PerOrderMatch {
        final String functionName; // null if unwrapped
        final ExprNode coefficient;
        final ExprNode stateFactor;

        PerOrderMatch(String functionName, ExprNode coefficient, ExprNode stateFactor) {
            this.functionName = functionName;
            this.coefficient = coefficient;
            this.stateFactor = stateFactor;
        }
    }

    private static PerOrderMatch classifyTerm(ExprNode term, int sign, ExprNode leaf) {
        // Try the identity case first: leaf is a plain multiplicative factor of the term.
        if (ExprAlgebra.onlyMultiplicativeAncestors(term, leaf)) {
            ExprNode coefficient = ExprAlgebra.applySign(ExprAlgebra.factorOut(term, leaf), sign);
            return new PerOrderMatch(null, coefficient, leaf);
        }

        // Otherwise, look for exactly one function call wrapping the leaf directly.
        ExprNode wrap = findImmediateFunctionWrap(term, leaf);
        if (wrap != null && ExprAlgebra.onlyMultiplicativeAncestors(term, wrap)) {
            ExprNode coefficient = ExprAlgebra.applySign(ExprAlgebra.factorOut(term, wrap), sign);
            return new PerOrderMatch(wrap.funcName, coefficient, wrap);
        }

        throw new IllegalArgumentException(
                "In term " + ExprAlgebra.describe(term) + ", y[" + leaf.stateIndex + "] is neither a plain "
                + "multiplicative factor nor wrapped in exactly one function call that is itself a plain "
                + "multiplicative factor. Supported shapes are A(t)*y[k] and A(t)*f(y[k]) only — not a power, "
                + "not a divisor, not nested inside more than one function, and not one of several arguments "
                + "to a multi-argument function.");
    }

    /**
     * Finds the OP node, if any, that is a single-argument function call
     * whose sole child is exactly (by reference) the given leaf — i.e. the
     * leaf's immediate parent is that one function call, nothing deeper.
     */
    private static ExprNode findImmediateFunctionWrap(ExprNode node, ExprNode leaf) {
        if (node.kind != ExprNode.Kind.OP) {
            return null;
        }
        if (node.funcName != null && node.children.size() == 1 && node.children.get(0) == leaf) {
            return node;
        }
        for (ExprNode child : node.children) {
            if (ExprAlgebra.containsNode(child, leaf)) {
                return findImmediateFunctionWrap(child, leaf);
            }
        }
        return null;
    }
}