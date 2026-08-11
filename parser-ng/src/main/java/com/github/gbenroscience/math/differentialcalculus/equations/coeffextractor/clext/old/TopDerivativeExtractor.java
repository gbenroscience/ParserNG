package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.old;

import java.util.ArrayList;
import java.util.List;

/**
 * Isolates the top-order derivative from a raw equation, requiring linearity
 * only in the single term that contains it — every other term may be
 * arbitrary, including nonlinear (y[k]^2, sin(y[j]), y[i]*y[j] for i != j,
 * and so on). This subsumes LinearFormExtractor's fully-linear case (that is
 * simply the special case where the remainder happens to decompose into
 * linear pieces) while additionally covering equations like the pendulum
 * (y[2] + sin(y[0]) = 0) that LinearFormExtractor must reject.
 *
 * Given A(t)*y[n] + remainder(t, Y) = g(t) — where remainder may reference
 * any y[0..n-1] and y[n] itself may appear nowhere else — this produces the
 * single divided top-derivative expression
 *
 *   y[n] = (g(t) - remainder(t, Y)) / A(t)
 *
 * ready to compile and hand to CompanionSystemHandles exactly like a
 * hand-written diffeqnHO lambda body would be.
 *
 * <h2>What this still cannot do</h2>
 * If y[n] itself is nonlinear in the equation — y[n]^2, sin(y[n]), y[n]
 * multiplied by another state variable — there is no general symbolic way to
 * isolate it, for the same reason a computer algebra system cannot always
 * "solve for x" from an arbitrary equation. extract() throws in that case,
 * naming the problem, rather than guessing. The two honest options at that
 * point: write the equation as a lambda (unrestricted, always available), or
 * treat it as a genuinely implicit equation and solve a residual via Newton
 * iteration each step — structurally close to what
 * DifferentialEquations.stepImplicitEulerCore's Newton core already does
 * internally, but not exposed as a front-end input shape today. That would
 * be separate, new work, not something this class attempts.
 */
public final class TopDerivativeExtractor {

    private TopDerivativeExtractor() {
    }

    public static final class Result {
        /** A(t) — the coefficient of the top-order term, y[order]. */
        public final ExprNode leadingCoefficient;
        /**
         * Everything else in the equation, signed and summed, moved to the
         * same side as the leading term (i.e. leadingCoefficient*y[order] +
         * remainder = 0 holds). May reference any y[0..order-1] and may be
         * nonlinear in them; never references y[order].
         */
        public final ExprNode remainder;
        /** The order of the equation — the top state index actually referenced. */
        public final int order;

        Result(ExprNode leadingCoefficient, ExprNode remainder, int order) {
            this.leadingCoefficient = leadingCoefficient;
            this.remainder = remainder;
            this.order = order;
        }

        /** Builds y[order] = -remainder / leadingCoefficient, ready to compile as the top-derivative expression. */
        public ExprNode topDerivativeExpression() {
            ExprNode negatedRemainder = ExprAlgebra.applySign(remainder, -1);
            return ExprNode.op('/', List.of(negatedRemainder, leadingCoefficient));
        }
    }

    /**
     * @param lhs the left-hand side of the equation as written
     * @param rhs the right-hand side of the equation as written (0 for a
     *            homogeneous equation written with "= 0")
     */
    public static Result extract(ExprNode lhs, ExprNode rhs) {
        ExprNode difference = ExprNode.op('-', List.of(lhs, rhs));
        List<ExprAlgebra.SignedTerm> terms = ExprAlgebra.collectAdditiveTerms(difference);

        // Pass 1: find the top state index referenced anywhere in the whole
        // equation — including inside nonlinear contexts, since we still need
        // to know the equation's order even where linearity isn't required.
        // Also detect (and validate) the single state-variable name used
        // across the whole equation, so every error message below reflects
        // the name actually used, whatever it is — not a hardcoded "y".
        List<ExprNode> allLeaves = new ArrayList<>();
        int topIndex = -1;
        for (ExprAlgebra.SignedTerm term : terms) {
            List<ExprNode> leaves = new ArrayList<>();
            ExprAlgebra.findStateLeaves(term.node, leaves);
            allLeaves.addAll(leaves);
            for (ExprNode leaf : leaves) {
                topIndex = Math.max(topIndex, leaf.stateIndex);
            }
        }
        if (topIndex < 0) {
            throw new IllegalArgumentException(
                    "Equation has no state-variable term at all — nothing to solve for.");
        }
        String stateVarName = ExprAlgebra.requireSingleStateVariableName(allLeaves);
        if (topIndex == 0) {
            throw new IllegalArgumentException(
                    "Equation only references " + stateVarName + "[0] — there is no derivative term. This "
                    + "framework solves differential equations, not algebraic ones; an order-0 equation has "
                    + "nothing to integrate.");
        }

        // Pass 2: find the (exactly one) leaf equal to stateVarName[topIndex],
        // confirm it occurs in exactly one term, confirm that term is linear
        // in it, and bundle every other term — regardless of its own
        // linearity — into the remainder untouched.
        ExprNode leadingTermNode = null;
        int leadingTermSign = 0;
        ExprNode topLeaf = null;
        int occurrencesOfTop = 0;
        List<ExprNode> remainderParts = new ArrayList<>();

        for (ExprAlgebra.SignedTerm term : terms) {
            List<ExprNode> leaves = new ArrayList<>();
            ExprAlgebra.findStateLeaves(term.node, leaves);

            ExprNode topLeafInThisTerm = null;
            for (ExprNode leaf : leaves) {
                if (leaf.stateIndex == topIndex) {
                    topLeafInThisTerm = leaf;
                    occurrencesOfTop++;
                }
            }

            if (topLeafInThisTerm != null) {
                if (leadingTermNode != null) {
                    throw new IllegalArgumentException(
                            "Equation is not solvable for " + stateVarName + "[" + topIndex + "]: it appears in "
                            + "more than one additive term. The top-order derivative must appear exactly once.");
                }
                leadingTermNode = term.node;
                leadingTermSign = term.sign;
                topLeaf = topLeafInThisTerm;
            } else {
                remainderParts.add(ExprAlgebra.applySign(term.node, term.sign));
            }
        }

        if (occurrencesOfTop != 1) {
            throw new IllegalArgumentException(
                    "Equation is not solvable for " + stateVarName + "[" + topIndex + "]: it appears "
                    + occurrencesOfTop + " times (e.g. " + stateVarName + "[" + topIndex + "]^2, or multiplied "
                    + "against itself). The top-order derivative must appear exactly once, as a plain factor. "
                    + "Write this equation as a lambda instead, or solve it as an implicit residual.");
        }
        if (!ExprAlgebra.onlyMultiplicativeAncestors(leadingTermNode, topLeaf)) {
            throw new IllegalArgumentException(
                    "Equation is not solvable for " + stateVarName + "[" + topIndex + "]: in term "
                    + ExprAlgebra.describe(leadingTermNode) + ", " + stateVarName + "[" + topIndex
                    + "] is not a plain multiplicative factor (it appears inside a power, a function call, "
                    + "or as a divisor). Write this equation as a lambda instead, or solve it as an implicit "
                    + "residual.");
        }

        ExprNode leadingCoefficient = ExprAlgebra.applySign(
                ExprAlgebra.factorOut(leadingTermNode, topLeaf), leadingTermSign);

        ExprNode remainder = remainderParts.isEmpty()
                ? ExprNode.number(0.0)
                : ExprAlgebra.sumAll(remainderParts);

        return new Result(leadingCoefficient, remainder, topIndex);
    }
}