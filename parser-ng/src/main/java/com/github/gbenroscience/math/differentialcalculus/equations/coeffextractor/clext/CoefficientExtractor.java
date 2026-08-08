package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Extracts per-derivative-order coefficients from an equation of the general
 * form ParserNG's diffeqn/diffeqnPath/diffeqnHO/diffeqnHOPath accept as their
 * first argument:
 *
 * <pre>
 *   A1(t)*y[n] + A2(t)*f1(y[n-1]) + A3(t)*f2(y[n-2]) + ... + Ak(t)*fn(y[0]) - B(t)
 * </pre>
 *
 * where each A_i(t) is a coefficient (function of t, or a constant), each f_i
 * is at most one wrapping function applied directly to that term's state
 * variable (identity — no wrap at all — is allowed and common; see -3*y[1] and
 * 3*x*y[0] in the worked example below), B(t) is an optional forcing term with
 * no state reference at all, and the equation is written with an implicit "= 0"
 * — there is no separate right-hand side argument to this class, unlike
 * LinearFormExtractor/TopDerivativeExtractor which pre-date this looser form.
 * The one required invariant, exactly as specified: <b>the top-order term
 * (A1(t)*y[n]) must always be strictly linear — no wrapping function is allowed
 * there</b>, even though every other order may have one.
 *
 * <h2>Three ways in</h2>
 * <ul>
 * <li>{@link #extract(String)} — a raw expression string, lexed here via
 * {@link Lexer} then parsed via {@link ExprParser}.</li>
 * <li>{@link #extract(List)} — an already-scanned flat token list, exactly the
 * form shown in ParserNG's own scanner output (one array entry per atomic
 * lexeme — parens, brackets, operators, digits, letters).</li>
 * <li>{@link #extract(ExprNode)} — an already-parsed expression tree, for
 * callers that have their own parser and just want the coefficient extraction
 * logic.</li>
 * </ul>
 * The first two both accept either the bare equation on its own, or the
 * <em>entire</em> scanned/raw {@code diffeqn(...)} call (equation plus t0, y0,
 * tEnd, method, etc.) — {@link ArgumentIsolator} detects which case it is and,
 * for the latter, isolates just the first argument before parsing. This is what
 * makes it safe to hand this class the literal output of ParserNG's own scanner
 * without pre-processing it yourself.
 *
 * <h2>The dependent variable's name is not fixed to "y"</h2>
 * The worked example below happens to use "y[...]", but nothing in this class
 * assumes that specific name. {@link #extract(String)} and
 * {@link #extract(List)} auto-detect whichever identifier is actually used with
 * bracket indexing in the equation (u[k], z[k], anything) and use it
 * consistently throughout the {@link Result} — error messages, the "absent
 * order" filler entries, all of it. If more than one distinct indexed name
 * appears in the same equation (almost certainly a mistake — an equation has
 * exactly one dependent variable), extraction fails immediately with a message
 * naming both. {@link Result#stateVariableName} exposes whichever name was
 * actually detected.
 * <p>
 * If a caller already knows what the name should be and wants that enforced up
 * front rather than discovered after the fact, {@link #extract(String, String)}
 * and {@link #extract(List, String)} accept an explicit required name — indexed
 * access on any other identifier is then rejected immediately during parsing,
 * with a message naming the mismatch.
 *
 * <h2>What counts as valid, precisely</h2>
 * The equation is split into additive terms (top-level + and -, exactly as in
 * LinearFormExtractor). Each term must reference at most one derivative order:
 * <ul>
 * <li><b>Zero state references</b> — the term is forcing-term (B(t)) material;
 * multiple such terms across the equation are summed.</li>
 * <li><b>Exactly one state reference, y[k]</b> — accepted if either (a) y[k] is
 * a plain multiplicative factor of the term (the identity case — no function
 * wraps it), or (b) y[k]'s immediate parent in the tree is a single-argument
 * function call (e.g. sin(y[k])), and that whole function-call node is itself a
 * plain multiplicative factor of the term. Anything deeper — two nested
 * functions, y[k] as an argument among several in a multi-arg call, y[k]^2,
 * y[k] as a divisor — is rejected with a specific reason, not silently
 * misclassified.</li>
 * <li><b>More than one state reference</b> (whether the same order twice, e.g.
 * y[2]*y[2], or two different orders, e.g. y[2]*y[1]) is rejected: this form
 * has no provision for coupling between orders.</li>
 * </ul>
 * If the same order k is found in more than one distinct additive term (e.g.
 * "sin(y[2]) + cos(y[2])"), that is also rejected — this form specifies exactly
 * one A_k(t)*f_k(y[n-k]) per order, so which of two competing terms should
 * "win" is genuinely ambiguous rather than something to guess at.
 *
 * Any order between 0 and the top order that never appears in the equation at
 * all is filled in with coefficient 0 (identity wrap) — the array and map
 * outputs are always dense across 0..topOrder, never sparse.
 *
 * <h2>Normalized coefficients — a separate, additive feature</h2>
 * Alongside the raw per-order coefficients above (which describe the equation
 * exactly as written), {@link Result} also exposes the equation with the top
 * derivative made the subject of the formula — i.e. divided through by the
 * leading coefficient A1(t), the same "isolate the top derivative" step
 * {@code LinearHODifferentialEquations.buildTopDerivative} performs. See {@link Result#normalizedCoefficients},
 * {@link Result#normalizedCoefficientsByOrder}, and
 * {@link Result#topDerivativeExpression} — none of the raw fields are affected
 * by this; both forms are always computed and available together.
 */
public final class CoefficientExtractor {

    private CoefficientExtractor() {
    }

    // ------------------------------------------------------------------
    // Result shape
    // ------------------------------------------------------------------
    /**
     * One derivative order's contribution, as it was actually written.
     */
    public static final class DerivativeTerm {

        public final int order;
        /**
         * null if this order was unwrapped (identity) or was absent entirely.
         */
        public final String functionName;
        /**
         * A_k(t), sign already folded in — coefficient * stateFactor
         * reconstructs the original term exactly.
         */
        public final ExprNode coefficient;
        /**
         * The bare y[order] leaf, or the whole f(y[order]) call if wrapped.
         */
        public final ExprNode stateFactor;
        /**
         * True if this order never actually appeared in the equation
         * (coefficient is a synthetic 0).
         */
        public final boolean absent;

        DerivativeTerm(int order, String functionName, ExprNode coefficient, ExprNode stateFactor, boolean absent) {
            this.order = order;
            this.functionName = functionName;
            this.coefficient = coefficient;
            this.stateFactor = stateFactor;
            this.absent = absent;
        }
    }

    public static final class Result {

        public final int topOrder;

        /**
         * The dependent variable's name as actually used in the equation (e.g.
         * "y", "u", "z") — detected automatically, or validated against an
         * explicitly required name if one was supplied to extract(...).
         */
        public final String stateVariableName;

        /**
         * Ordered highest order first: terms.get(0) is order topOrder,
         * terms.get(topOrder) is order 0.
         */
        public final List<DerivativeTerm> terms;

        /**
         * B(t), or null if the equation had no state-free term at all.
         */
        public final ExprNode forcingOrNull;

        /**
         * Keyed by derivative order k (0..topOrder); every key present.
         */
        public final Map<Integer, ExprNode> coefficientsByOrder;

        /**
         * Parallel to derivativeTerms below, highest order first, length
         * topOrder+1.
         */
        public final ExprNode[] coefficients;

        /**
         * Parallel to coefficients above. coefficients[i] * derivativeTerms[i]
         * reconstructs the i-th term of the equation exactly (sign included in
         * coefficients[i]) — this is the array pairing requested: same order,
         * corresponding entries multiply back to the original term.
         */
        public final ExprNode[] derivativeTerms;

        /**
         * Parallel to the two arrays above; null entry means that order was
         * unwrapped (identity).
         */
        public final String[] wrappingFunctionNames;

        // ------------------------------------------------------------------
        // Normalized form: the equation with the top derivative made the
        // subject — i.e. divided through by the leading coefficient A1(t).
        // A SEPARATE, ADDITIVE feature: none of the fields above are
        // affected or replaced by this.
        // ------------------------------------------------------------------
        /**
         * Keyed by derivative order k (0..topOrder-1) — the top order itself
         * has no entry here, since after normalization it is alone on the
         * left-hand side (y[topOrder] = ...), not part of the right-hand side
         * being described. Each value is -(coefficients[k] /
         * coefficients[topOrder]) — the sign flip is because moving a term from
         * one side of "= 0" to the other side negates it.
         */
        public final Map<Integer, ExprNode> normalizedCoefficientsByOrder;

        /**
         * Parallel to normalizedDerivativeTerms below, highest non-top order
         * first, length topOrder (not topOrder+1).
         */
        public final ExprNode[] normalizedCoefficients;

        /**
         * Parallel to normalizedCoefficients above — the same stateFactor (bare
         * y[k] or wrapped f(y[k])) as in derivativeTerms, just excluding the
         * top order's slot. normalizedCoefficients[i] *
         * normalizedDerivativeTerms[i] reconstructs the i-th right-hand-side
         * summand exactly (sign included).
         */
        public final ExprNode[] normalizedDerivativeTerms;

        /**
         * B(t) / A1(t), or null if the equation had no forcing term.
         */
        public final ExprNode normalizedForcingOrNull;

        /**
         * The fully assembled right-hand side: y[topOrder] =
         * topDerivativeExpression. This is the sum of every
         * normalizedCoefficients[i]*normalizedDerivativeTerms[i], plus
         * normalizedForcingOrNull if present — ready to compile and hand to
         * CompanionSystemHandles/HigherOrderODE exactly like a hand-written
         * diffeqnHO lambda body, the same role
         * TopDerivativeExtractor.Result.topDerivativeExpression() plays for
         * that extractor.
         */
        public final ExprNode topDerivativeExpression;
        
        /**
         * 
         * @param topOrder
         * @param stateVariableName
         * @param terms
         * @param forcingOrNull 
         */
        Result(int topOrder, String stateVariableName, List<DerivativeTerm> terms, ExprNode forcingOrNull) {
            this.topOrder = topOrder;
            this.stateVariableName = stateVariableName;
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

            // terms.get(0) is always the top order (list is built highest-first).
            ExprNode leadingCoefficient = terms.get(0).coefficient;

            int rhsCount = terms.size() - 1; // every order except the top
            this.normalizedCoefficientsByOrder = new LinkedHashMap<>();
            this.normalizedCoefficients = new ExprNode[rhsCount];
            this.normalizedDerivativeTerms = new ExprNode[rhsCount];

            List<ExprNode> rhsSummands = new ArrayList<>();
            for (int i = 1; i < terms.size(); i++) {
                DerivativeTerm dt = terms.get(i);
                ExprNode ratio = ExprNode.op('/', List.of(dt.coefficient, leadingCoefficient));
                ExprNode normalized = ExprAlgebra.applySign(ratio, -1);

                int arrIndex = i - 1;
                normalizedCoefficients[arrIndex] = normalized;
                normalizedDerivativeTerms[arrIndex] = dt.stateFactor;
                normalizedCoefficientsByOrder.put(dt.order, normalized);

                rhsSummands.add(ExprNode.op('*', List.of(normalized, dt.stateFactor)));
            }

            this.normalizedForcingOrNull = forcingOrNull == null
                    ? null
                    : ExprNode.op('/', List.of(forcingOrNull, leadingCoefficient));
            if (normalizedForcingOrNull != null) {
                rhsSummands.add(normalizedForcingOrNull);
            }

            this.topDerivativeExpression = rhsSummands.isEmpty()
                    ? ExprNode.number(0.0)
                    : ExprAlgebra.sumAll(rhsSummands);
        }
    }

    /**
     * Entry points
     *
     * @param rawExpressionOrFullCall
     * @return
     */
    public static Result extract(String rawExpressionOrFullCall) {
        return extract(Lexer.tokenize(rawExpressionOrFullCall));
    }

    /**
     * Same as {@link #extract(String)}, but requires the dependent variable to
     * be exactly requiredStateVarName — an indexed reference to any other
     * identifier is rejected immediately during parsing, with a message naming
     * the mismatch, rather than after full extraction.
     *
     * @param rawExpressionOrFullCall
     * @param requiredStateVarName
     * @return
     */
    public static Result extract(String rawExpressionOrFullCall, String requiredStateVarName) {
        return extract(Lexer.tokenize(rawExpressionOrFullCall), requiredStateVarName);
    }

    /**
     *
     * @param scannedTokensOrFullCall
     * @return
     */
    public static Result extract(List<String> scannedTokensOrFullCall) {
        List<String> equationTokens = ArgumentIsolator.isolateFirstArgument(scannedTokensOrFullCall);
        ExprNode expression = ExprParser.parse(equationTokens);
        return extract(expression);
    }

    /**
     * Same as {@link #extract(List)}, but requires the dependent variable to be
     * exactly requiredStateVarName.
     *
     * @param scannedTokensOrFullCall
     * @param requiredStateVarName
     * @return
     */
    public static Result extract(List<String> scannedTokensOrFullCall, String requiredStateVarName) {
        List<String> equationTokens = ArgumentIsolator.isolateFirstArgument(scannedTokensOrFullCall);
        ExprNode expression = ExprParser.parse(equationTokens, requiredStateVarName);
        return extract(expression);
    }

    /**
     * @param expression the whole equation, implicitly "expression = 0" —
     * matching the form's own convention of folding any right-hand side in as a
     * subtracted term
     */
    public static Result extract(ExprNode expression) {
        List<ExprAlgebra.SignedTerm> terms = ExprAlgebra.collectAdditiveTerms(expression);

        // Detect (and validate) the single state-variable name used across the
        // whole equation up front — whatever it actually is, not assumed to be
        // "y". If extract(..., requiredStateVarName) routed through ExprParser's
        // restrictive mode, every leaf here already carries that one name; this
        // still catches the case of an ExprNode built some other way (e.g. by a
        // caller's own parser) that happens to mix two different names.
        List<ExprNode> allLeavesForNameCheck = new ArrayList<>();
        for (ExprAlgebra.SignedTerm term : terms) {
            ExprAlgebra.findStateLeaves(term.node, allLeavesForNameCheck);
        }
        String stateVarName = ExprAlgebra.requireSingleStateVariableName(allLeavesForNameCheck);

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
                        stateVarName + "[" + leaf.stateIndex + "] appears in more than one additive term — "
                        + "this form specifies exactly one coefficient (and at most one wrapping function) "
                        + "per order.");
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
                    "The top-order term " + stateVarName + "[" + topOrder + "] is wrapped in " + topMatch.functionName
                    + "(...) — the top derivative must always be strictly linear (A1(t)*" + stateVarName + "["
                    + topOrder + "], no wrapping function), per this form's own rule.");
        }

        List<DerivativeTerm> ordered = new ArrayList<>();
        for (int k = topOrder; k >= 0; k--) {
            PerOrderMatch m = matchesByOrder.get(k);
            if (m == null) {
                ordered.add(new DerivativeTerm(k, null, ExprNode.number(0.0), ExprNode.stateVariable(stateVarName, k), true));
            } else {
                ordered.add(new DerivativeTerm(k, m.functionName, m.coefficient, m.stateFactor, false));
            }
        }

        ExprNode forcing = forcingParts.isEmpty() ? null : ExprAlgebra.applySign(ExprAlgebra.sumAll(forcingParts), -1);

        return new Result(topOrder, stateVarName, ordered, forcing);
    }

    /**
     * Per-term classification: identity or exactly-one-function-wrap
     */
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

    /**
     *
     * @param term
     * @param sign
     * @param leaf
     * @return
     */
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
                "In term " + ExprAlgebra.describe(term) + ", " + leaf.variableName + "[" + leaf.stateIndex
                + "] is neither a plain multiplicative factor nor wrapped in exactly one function call that "
                + "is itself a plain multiplicative factor. Supported shapes are A(t)*" + leaf.variableName
                + "[k] and A(t)*f(" + leaf.variableName + "[k]) only — not a power, not a divisor, not nested "
                + "inside more than one function, and not one of several arguments to a multi-argument function.");
    }

    /**
     * Finds the OP node, if any, that is a single-argument function call whose
     * sole child is exactly (by reference) the given leaf — i.e. the leaf's
     * immediate parent is that one function call, nothing deeper.
     *
     * @param node
     * @param leaf
     * @return
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
