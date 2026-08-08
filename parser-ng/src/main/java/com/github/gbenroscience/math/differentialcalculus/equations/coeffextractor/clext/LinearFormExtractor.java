package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Extracts the coefficient list and forcing term from a raw linear n-th
 * order ODE written as an equation:
 *
 *   A1(t)*y[n] + A2(t)*y[n-1] + ... + A(n+1)(t)*y[0] = g(t)
 *
 * where the A's and g are functions of t (or constants) and the y[k] terms
 * are exactly the state derivatives, each appearing linearly.
 *
 * This is the STRICT extractor — every additive term in the equation must be
 * linear in exactly one y[k]. For the relaxed case where only the top-order
 * term y[n] needs to be linear and everything else may be arbitrary
 * (including nonlinear), see TopDerivativeExtractor instead.
 *
 * Output feeds directly into the existing
 * LinearHODifferentialEquations.buildTopDerivative — this class does no
 * numerical work and produces no MethodHandle/ODEFunction itself, only the
 * coefficient sub-expressions (as ExprNode) that ParserNG's ordinary
 * expression compiler would then compile the normal way, exactly as it
 * already compiles any other sub-expression.
 *
 * <h2>Open dependency on ParserNG's real grammar</h2>
 * This class assumes y[k] is identifiable as a leaf ExprNode with
 * isStateVariable() true and stateIndex == k. How that gets populated from
 * ParserNG's actual indexed-variable syntax (Token kind, MATRIX access, or
 * whatever y[k] actually lowers to) is the one piece not shown here — see
 * ExprNode's javadoc. Everything past that point is grammar-independent.
 *
 * <h2>What counts as linear, precisely</h2>
 * A term (one additive summand of the equation, after moving everything to
 * one side) is accepted as the y[k] term if and only if:
 * <ol>
 *   <li>it contains exactly one state-variable leaf, y[k] for some k;</li>
 *   <li>every operator on the path from that leaf up to the term's root is
 *       multiplication or division with the state variable as a numerator
 *       factor (never a divisor, never inside pow/sin/cos/etc, never
 *       appearing a second time in the same term).</li>
 * </ol>
 * A term with zero state-variable leaves is forcing-term material. A term
 * that fails both — most commonly two different y[k]'s multiplied together,
 * a squared or transcendental state reference, or a state variable used as a
 * divisor — throws IllegalArgumentException naming the term and the reason,
 * rather than silently misclassifying it. There is no best-effort fallback
 * here on purpose: a wrong coefficient produces a solver that runs and
 * converges to a wrong answer with no diagnostic, which is worse than
 * refusing to compile.
 */
public final class LinearFormExtractor {

    private LinearFormExtractor() {
    }

    public static final class Result {
        /** Length order+1, highest order first: coefficients.get(0) is A1 (the y[n] coefficient). */
        public final List<ExprNode> coefficients;
        /** g(t), or null for the homogeneous case (nothing on the state-independent side). */
        public final ExprNode forcingOrNull;
        public final int order;

        Result(List<ExprNode> coefficients, ExprNode forcingOrNull, int order) {
            this.coefficients = coefficients;
            this.forcingOrNull = forcingOrNull;
            this.order = order;
        }
    }

    /**
     * @param lhs the left-hand side of the equation as written
     * @param rhs the right-hand side of the equation as written (0 for a
     *            homogeneous equation written with "= 0")
     */
    public static Result extract(ExprNode lhs, ExprNode rhs) {
        // Move everything to one side: difference = lhs - rhs = 0.
        ExprNode difference = ExprNode.op('-', List.of(lhs, rhs));
        List<ExprAlgebra.SignedTerm> terms = ExprAlgebra.collectAdditiveTerms(difference);

        // Detect (and validate) the single state-variable name used across the whole
        // equation up front, so every error message below reflects the name actually
        // used, whatever it is — not a hardcoded "y".
        List<ExprNode> allLeaves = new ArrayList<>();
        for (ExprAlgebra.SignedTerm term : terms) {
            ExprAlgebra.findStateLeaves(term.node, allLeaves);
        }
        ExprAlgebra.requireSingleStateVariableName(allLeaves);

        Map<Integer, List<ExprNode>> coefficientPartsByIndex = new HashMap<>();
        List<ExprNode> forcingParts = new ArrayList<>();
        int maxIndexSeen = -1;

        for (ExprAlgebra.SignedTerm term : terms) {
            Classification c = classify(term.node);
            ExprNode signedCoefficient = ExprAlgebra.applySign(c.remainder, term.sign);

            if (c.stateIndex == null) {
                // No state reference at all -> belongs on the forcing side,
                // with sign flipped since it started on the state-equation
                // side (difference = lhs - rhs, forcing side is -difference's
                // state-free part).
                forcingParts.add(ExprAlgebra.applySign(signedCoefficient, -1));
            } else {
                coefficientPartsByIndex
                        .computeIfAbsent(c.stateIndex, k -> new ArrayList<>())
                        .add(signedCoefficient);
                maxIndexSeen = Math.max(maxIndexSeen, c.stateIndex);
            }
        }

        if (maxIndexSeen < 0) {
            throw new IllegalArgumentException(
                    "Equation has no state-variable term at all — nothing to solve for.");
        }

        // y[0]..y[maxIndexSeen-1] are the companion state components; y[maxIndexSeen] is the top derivative
        // being solved for and is NOT itself a state component — this matches y0.length == order used
        // throughout CompanionSystemHandles/HigherOrderODE. Highest order first: coefficients.get(0) is for
        // y[maxIndexSeen] (the top derivative), down to coefficients.get(maxIndexSeen) for y[0].
        List<ExprNode> coefficients = new ArrayList<>();
        for (int k = maxIndexSeen; k >= 0; k--) {
            List<ExprNode> parts = coefficientPartsByIndex.get(k);
            if (parts == null || parts.isEmpty()) {
                coefficients.add(ExprNode.number(0.0)); // y[k] never appeared -> coefficient 0
            } else {
                coefficients.add(ExprAlgebra.sumAll(parts));
            }
        }

        ExprNode forcing = forcingParts.isEmpty() ? null : ExprAlgebra.sumAll(forcingParts);

        return new Result(coefficients, forcing, maxIndexSeen);
    }

    // ------------------------------------------------------------------
    // Classify one term
    // ------------------------------------------------------------------

    private static final class Classification {
        final Integer stateIndex; // null if this term has no state reference
        final ExprNode remainder; // the term with the state leaf factored out (== the term itself if stateIndex is null)

        Classification(Integer stateIndex, ExprNode remainder) {
            this.stateIndex = stateIndex;
            this.remainder = remainder;
        }
    }

    private static Classification classify(ExprNode term) {
        List<ExprNode> stateLeaves = new ArrayList<>();
        ExprAlgebra.findStateLeaves(term, stateLeaves);

        if (stateLeaves.isEmpty()) {
            return new Classification(null, term);
        }
        if (stateLeaves.size() > 1) {
            throw new IllegalArgumentException(
                    "Equation is not linear: term references the state more than once ("
                    + ExprAlgebra.describe(term) + "). Each additive term must contain exactly one y[k].");
        }

        ExprNode stateLeaf = stateLeaves.get(0);
        if (!ExprAlgebra.onlyMultiplicativeAncestors(term, stateLeaf)) {
            throw new IllegalArgumentException(
                    "Equation is not linear: " + stateLeaf.variableName + "[" + stateLeaf.stateIndex
                    + "] in term " + ExprAlgebra.describe(term)
                    + " is not a plain multiplicative factor (it appears inside a power, a function call, "
                    + "or as a divisor). Only A(t)*" + stateLeaf.variableName + "[k] — a coefficient times "
                    + "a state variable — is supported.");
        }

        ExprNode remainder = ExprAlgebra.factorOut(term, stateLeaf);
        return new Classification(stateLeaf.stateIndex, remainder);
    }
}