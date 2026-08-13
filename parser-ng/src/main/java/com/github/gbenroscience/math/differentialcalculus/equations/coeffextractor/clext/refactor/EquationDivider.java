package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.refactor;

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.ExprNode;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.TokenTreeBuilder;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.CanonicalFrame;
import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The representation-agnostic half of coefficient extraction — term
 * splitting, linearity checking against the top-order state, coefficient
 * division, and the canonical/real frame mapping. None of this touches
 * {@code ODEFunction} or {@code MethodHandle}; it only builds an {@link
 * ExprNode} tree. Factored out of {@link CoefficientExtractor} so the Turbo
 * tier ({@code TurboCoefficientExtractor}) can share it exactly rather than
 * duplicating ~150 lines of identical symbolic logic — the two extractors
 * differ only in the one line that compiles the resulting tree ({@code
 * ExprNodeCompiler.compileStandard} vs {@code compileTurbo}).
 *
 * <h2>Algorithm</h2>
 * Two passes over the tree {@link TokenTreeBuilder} builds from the isolated
 * postfix:
 * <ol>
 *   <li><b>Split into additive terms.</b> {@link #collectTerms} recursively
 *       walks every top-level {@code +}/binary {@code -}/unary {@code -},
 *       carrying an accumulated sign; anything else (a product, a function
 *       call, a bare variable) is one term, not split further.</li>
 *   <li><b>Classify each term against the top-order state {@code y[order]}.</b>
 *       {@link #countStateOccurrences} counts how many times it appears:
 *       <ul>
 *         <li><b>Zero</b> — independent of {@code y[order]}, goes into the
 *             remainder (which may be arbitrarily nonlinear in every OTHER
 *             state component — {@code sin(y[0])} is exactly this case).</li>
 *         <li><b>More than one</b> — rejected outright.</li>
 *         <li><b>Exactly one</b> — {@link #substituteIfLinear} walks from
 *             the term's root down to that leaf, requiring every operator on
 *             the path to be {@code *} (either side), {@code /} (numerator
 *             only), or unary {@code -} (linear — equivalent to multiplying
 *             by -1). Anything else on that path fails the extraction rather
 *             than guessing. On success the leaf is replaced with {@code 1}.</li>
 *       </ul>
 *       Multiple terms may independently be linear in {@code y[order]}
 *       (e.g. {@code 2*y[3] + 3*y[3]}); their coefficients are summed.</li>
 * </ol>
 * The divided result is {@code y[order] = -remainder / topCoefficientSum}.
 *
 * <h2>The frame-ordering fix</h2>
 * {@link #buildCanonicalToReal} scans the ORIGINAL (undivided) tree for
 * every {@code y[k]} leaf's real, VariableRegistry-assigned frame index —
 * in whatever order they happened to appear in the source text — and
 * reports them keyed by canonical index instead ({@code canonicalToReal[1+k]}).
 * A state index that never appears anywhere in the text gets {@link
 * CanonicalFrame#NO_REAL_SLOT} rather than a fabricated frame index.
 */
public final class EquationDivider {

    private EquationDivider() {
    }

    /** Result of dividing an equation: the y[order] = ... tree, plus its real-frame mapping. */
   public static final class Divided {
        public final ExprNode tree;
        public final int[] canonicalToReal;
        public final int realFrameSize;

        Divided(ExprNode tree, int[] canonicalToReal, int realFrameSize) {
            this.tree = tree;
            this.canonicalToReal = canonicalToReal;
            this.realFrameSize = realFrameSize;
        }
 
        
        
    }

    public static Divided divide(Token[] equationPostfix, int order) {
        ExprNode root = TokenTreeBuilder.fromPostfix(equationPostfix);

        List<TermWithSign> terms = new ArrayList<>();
        collectTerms(root, false, terms);

        List<ExprNode> topTerms = new ArrayList<>();
        List<ExprNode> remainderTerms = new ArrayList<>();
        for (TermWithSign t : terms) {
            int topCount = countStateOccurrences(t.term, order);
            if (topCount == 0) {
                remainderTerms.add(signed(t.term, t.negative));
                continue;
            }
            ExprNode coefficient = extractLinearCoefficient(t.term, order);
            topTerms.add(signed(coefficient, t.negative));
        }
        if (topTerms.isEmpty()) {
            throw new IllegalArgumentException(
                    "Equation never references the top-order state y[" + order + "] -- check that the "
                    + "equation's highest derivative matches y0's length (order).");
        }

        ExprNode topCoefficientSum = sumSigned(topTerms);
        ExprNode remainderSum = sumSigned(remainderTerms);
        ExprNode negatedRemainder = remainderSum == null
                ? ExprNode.number(0.0) : ExprNode.op('-', Arrays.asList(remainderSum));
        ExprNode divided = ExprNode.op('/', Arrays.asList(negatedRemainder, topCoefficientSum));

        int[] canonicalToReal = buildCanonicalToReal(root, order);
        int realFrameSize = computeRealFrameSize(canonicalToReal);

        return new Divided(divided, canonicalToReal, realFrameSize);
    }

    // ------------------------------------------------------------------
    // Term splitting
    // ------------------------------------------------------------------

    private static final class TermWithSign {
        final ExprNode term;
        final boolean negative;

        TermWithSign(ExprNode term, boolean negative) {
            this.term = term;
            this.negative = negative;
        }
    }

    private static void collectTerms(ExprNode node, boolean negate, List<TermWithSign> out) {
        if (node.kind == ExprNode.Kind.OP && node.funcName == null) {
            if (node.isOp('+') && node.children.size() == 2) {
                collectTerms(node.children.get(0), negate, out);
                collectTerms(node.children.get(1), negate, out);
                return;
            }
            if (node.isOp('-') && node.children.size() == 2) {
                collectTerms(node.children.get(0), negate, out);
                collectTerms(node.children.get(1), !negate, out);
                return;
            }
            if (node.isOp('-') && node.children.size() == 1) {
                collectTerms(node.children.get(0), !negate, out);
                return;
            }
        }
        out.add(new TermWithSign(node, negate));
    }

    private static ExprNode signed(ExprNode term, boolean negative) {
        return negative ? ExprNode.op('-', Arrays.asList(term)) : term;
    }

    private static ExprNode sumSigned(List<ExprNode> terms) {
        if (terms.isEmpty()) {
            return null;
        }
        ExprNode acc = terms.get(0);
        for (int i = 1; i < terms.size(); i++) {
            acc = ExprNode.op('+', Arrays.asList(acc, terms.get(i)));
        }
        return acc;
    }

    // ------------------------------------------------------------------
    // Linearity checking / coefficient extraction
    // ------------------------------------------------------------------

    private static int countStateOccurrences(ExprNode node, int stateIndex) {
        if (node.kind == ExprNode.Kind.VARIABLE && node.isStateVariable() && node.stateIndex == stateIndex) {
            return 1;
        }
        if (node.kind == ExprNode.Kind.OP) {
            int total = 0;
            for (ExprNode child : node.children) {
                total += countStateOccurrences(child, stateIndex);
            }
            return total;
        }
        return 0;
    }

    private static ExprNode extractLinearCoefficient(ExprNode term, int stateIndex) {
        int count = countStateOccurrences(term, stateIndex);
        if (count > 1) {
            throw new IllegalArgumentException(
                    "Equation is not linear in y[" + stateIndex + "]: it appears " + count
                    + " times within a single term.");
        }
        ExprNode result = substituteIfLinear(term, stateIndex);
        if (result == null) {
            throw new IllegalArgumentException(
                    "Equation is not linear in y[" + stateIndex + "]: it does not appear as a plain "
                    + "multiplicative factor (found inside a power, a function call, or as a divisor).");
        }
        return result;
    }

    /**
     * Returns {@code node} with the (known-single, by {@link
     * #extractLinearCoefficient}'s precondition) {@code y[stateIndex]} leaf
     * replaced by {@code 1}, provided every operator between {@code node}
     * and that leaf is {@code *}, {@code /} (numerator side only), or unary
     * {@code -}. Returns {@code null} the moment the path is disqualified.
     */
    private static ExprNode substituteIfLinear(ExprNode node, int stateIndex) {
        if (node.kind == ExprNode.Kind.VARIABLE) {
            if (node.isStateVariable() && node.stateIndex == stateIndex) {
                return ExprNode.number(1.0);
            }
            return node;
        }
        if (node.kind == ExprNode.Kind.NUMBER) {
            return node;
        }
        if (node.isFunctionCall()) {
            return countStateOccurrences(node, stateIndex) == 0 ? node : null;
        }
        if (node.isOp('*') && node.children.size() == 2) {
            ExprNode left = node.children.get(0);
            ExprNode right = node.children.get(1);
            boolean leftHas = countStateOccurrences(left, stateIndex) > 0;
            boolean rightHas = countStateOccurrences(right, stateIndex) > 0;
            if (leftHas && rightHas) {
                return null;
            }
            if (leftHas) {
                ExprNode newLeft = substituteIfLinear(left, stateIndex);
                return newLeft == null ? null : ExprNode.op('*', Arrays.asList(newLeft, right));
            }
            if (rightHas) {
                ExprNode newRight = substituteIfLinear(right, stateIndex);
                return newRight == null ? null : ExprNode.op('*', Arrays.asList(left, newRight));
            }
            return node;
        }
        if (node.isOp('/') && node.children.size() == 2) {
            ExprNode numerator = node.children.get(0);
            ExprNode denominator = node.children.get(1);
            if (countStateOccurrences(denominator, stateIndex) > 0) {
                return null; // the state variable may never be a divisor
            }
            if (countStateOccurrences(numerator, stateIndex) > 0) {
                ExprNode newNumerator = substituteIfLinear(numerator, stateIndex);
                return newNumerator == null ? null : ExprNode.op('/', Arrays.asList(newNumerator, denominator));
            }
            return node;
        }
        if (node.isOp('-') && node.children.size() == 1) {
            // Unary minus is linear (equivalent to multiplying by -1), so it's
            // treated like a multiplicative ancestor rather than disqualifying
            // the path -- e.g. "t*(-y[2])" extracts coefficient "t*(-1)".
            ExprNode child = node.children.get(0);
            if (countStateOccurrences(child, stateIndex) == 0) {
                return node;
            }
            ExprNode newChild = substituteIfLinear(child, stateIndex);
            return newChild == null ? null : ExprNode.op('-', Arrays.asList(newChild));
        }
        // Any other operator (binary '+', binary '-', '^') -- if the target
        // state variable is anywhere underneath, the path is disqualified;
        // otherwise this subtree is irrelevant to the substitution and
        // passes through unchanged.
        return countStateOccurrences(node, stateIndex) == 0 ? node : null;
    }

    // ------------------------------------------------------------------
    // Frame mapping -- the fix for the frame-ordering bug
    // ------------------------------------------------------------------

    private static int[] buildCanonicalToReal(ExprNode root, int order) {
        int[] canonicalToReal = new int[1 + order];
        Arrays.fill(canonicalToReal, CanonicalFrame.NO_REAL_SLOT);
        String[] independentVarName = new String[1];
        collectFrameIndices(root, order, canonicalToReal, independentVarName);
        return canonicalToReal;
    }

    private static void collectFrameIndices(ExprNode node, int order,
                                             int[] canonicalToReal, String[] independentVarName) {
        if (node.kind == ExprNode.Kind.VARIABLE) {
            if (node.isStateVariable()) {
                int k = node.stateIndex;
                if (k >= 0 && k < order) {
                    canonicalToReal[1 + k] = node.frameIndex;
                }
                // k == order is the top-order state being solved for -- it has
                // no canonical slot (it isn't part of the divided expression
                // once substituted out), so it's simply not recorded here.
                return;
            }
            if (independentVarName[0] == null) {
                independentVarName[0] = node.variableName;
                canonicalToReal[0] = node.frameIndex;
            } else if (!independentVarName[0].equals(node.variableName)) {
                throw new IllegalArgumentException(
                        "Equation references more than one independent variable ('" + independentVarName[0]
                        + "' and '" + node.variableName + "') -- expected exactly one.");
            }
            return;
        }
        if (node.kind == ExprNode.Kind.OP) {
            for (ExprNode child : node.children) {
                collectFrameIndices(child, order, canonicalToReal, independentVarName);
            }
        }
    }

    private static int computeRealFrameSize(int[] canonicalToReal) {
        int max = 0;
        for (int slot : canonicalToReal) {
            if (slot != CanonicalFrame.NO_REAL_SLOT) {
                max = Math.max(max, slot + 1);
            }
        }
        return Math.max(max, 1);
    }
}