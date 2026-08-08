package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

import java.util.ArrayList;
import java.util.List;

/**
 * Shared low-level tree operations used by both LinearFormExtractor (strict:
 * every term must be linear) and TopDerivativeExtractor (relaxed: only the
 * single term containing the highest-order state reference must be linear;
 * everything else may be arbitrary, including nonlinear).
 *
 * Package-private: an internal detail of the extraction algorithms, not part
 * of the public surface either extractor exposes.
 */
final class ExprAlgebra {

    private ExprAlgebra() {
    }

    // ------------------------------------------------------------------
    // Split into signed additive terms
    // ------------------------------------------------------------------

    static final class SignedTerm {
        final int sign;
        final ExprNode node;

        SignedTerm(int sign, ExprNode node) {
            this.sign = sign;
            this.node = node;
        }
    }

    static List<SignedTerm> collectAdditiveTerms(ExprNode root) {
        List<SignedTerm> out = new ArrayList<>();
        collectAdditiveTerms(root, 1, out);
        return out;
    }

    private static void collectAdditiveTerms(ExprNode node, int sign, List<SignedTerm> out) {
        if (node.isOp('+')) {
            collectAdditiveTerms(node.children.get(0), sign, out);
            collectAdditiveTerms(node.children.get(1), sign, out);
        } else if (node.isOp('-') && node.children.size() == 2) {
            collectAdditiveTerms(node.children.get(0), sign, out);
            collectAdditiveTerms(node.children.get(1), -sign, out);
        } else if (node.isOp('-') && node.children.size() == 1) {
            // unary minus
            collectAdditiveTerms(node.children.get(0), -sign, out);
        } else {
            out.add(new SignedTerm(sign, node));
        }
    }

    // ------------------------------------------------------------------
    // Locating and factoring out a state-variable leaf
    // ------------------------------------------------------------------

    static void findStateLeaves(ExprNode node, List<ExprNode> out) {
        if (node.isStateVariable()) {
            out.add(node);
            return;
        }
        if (node.kind == ExprNode.Kind.OP) {
            for (ExprNode child : node.children) {
                findStateLeaves(child, out);
            }
        }
    }

    /**
     * True iff every operator between the term's root and the given leaf is
     * multiplication, or division where the leaf sits on the numerator side
     * — i.e. the leaf is a plain multiplicative factor of the term, never
     * inside a power, a function call, an addition/subtraction, or a
     * denominator.
     */
    static boolean onlyMultiplicativeAncestors(ExprNode node, ExprNode target) {
        if (node == target) {
            return true;
        }
        if (node.kind != ExprNode.Kind.OP) {
            return false; // target not found under a non-OP leaf (shouldn't happen if target is truly inside node)
        }
        if (node.isOp('*')) {
            for (ExprNode child : node.children) {
                if (containsNode(child, target)) {
                    return onlyMultiplicativeAncestors(child, target);
                }
            }
        }
        if (node.isOp('/')) {
            ExprNode numerator = node.children.get(0);
            ExprNode denominator = node.children.get(1);
            if (containsNode(numerator, target)) {
                return onlyMultiplicativeAncestors(numerator, target);
            }
            if (containsNode(denominator, target)) {
                return false; // state variable as a divisor -> not a plain multiplicative factor
            }
        }
        return false; // any other operator (+, -, ^, function call) on the path -> not a plain multiplicative factor
    }

    static boolean containsNode(ExprNode node, ExprNode target) {
        if (node == target) {
            return true;
        }
        if (node.kind != ExprNode.Kind.OP) {
            return false;
        }
        for (ExprNode child : node.children) {
            if (containsNode(child, target)) {
                return true;
            }
        }
        return false;
    }

    /** Rebuilds the term with the state leaf replaced by 1 — i.e. "all the other multiplicative factors". */
    static ExprNode factorOut(ExprNode node, ExprNode target) {
        if (node == target) {
            return ExprNode.number(1.0);
        }
        if (node.isOp('*')) {
            ExprNode a = node.children.get(0);
            ExprNode b = node.children.get(1);
            return containsNode(a, target)
                    ? multiplyOut(factorOut(a, target), b)
                    : multiplyOut(a, factorOut(b, target));
        }
        if (node.isOp('/')) {
            ExprNode numerator = node.children.get(0);
            ExprNode denominator = node.children.get(1);
            return ExprNode.op('/', List.of(factorOut(numerator, target), denominator));
        }
        return node; // unreachable given onlyMultiplicativeAncestors already passed
    }

    private static ExprNode multiplyOut(ExprNode a, ExprNode b) {
        if (a.kind == ExprNode.Kind.NUMBER && a.numberValue == 1.0) {
            return b;
        }
        if (b.kind == ExprNode.Kind.NUMBER && b.numberValue == 1.0) {
            return a;
        }
        return ExprNode.op('*', List.of(a, b));
    }

    // ------------------------------------------------------------------
    // Sign and summation helpers
    // ------------------------------------------------------------------

    static ExprNode applySign(ExprNode node, int sign) {
        return sign >= 0 ? node : ExprNode.op('-', List.of(node));
    }

    /** Sums an already-signed list of terms (each element must already have applySign applied if needed). */
    static ExprNode sumAll(List<ExprNode> parts) {
        ExprNode acc = parts.get(0);
        for (int i = 1; i < parts.size(); i++) {
            acc = ExprNode.op('+', List.of(acc, parts.get(i)));
        }
        return acc;
    }

    // ------------------------------------------------------------------
    // State-variable name consistency
    // ------------------------------------------------------------------

    /**
     * Confirms every given state-variable leaf shares the same
     * variableName, returning it (or null if the list is empty). Throws if
     * more than one distinct name is found — an equation is expected to
     * have exactly one dependent variable, so e.g. accidentally mixing
     * y[2] and u[1] in the same equation is a real error worth catching
     * here rather than letting it silently produce an undefined result.
     */
    static String requireSingleStateVariableName(List<ExprNode> stateLeaves) {
        String name = null;
        for (ExprNode leaf : stateLeaves) {
            if (name == null) {
                name = leaf.variableName;
            } else if (!name.equals(leaf.variableName)) {
                throw new IllegalArgumentException(
                        "Equation references more than one distinct state-variable name ('" + name
                        + "' and '" + leaf.variableName + "') — only one dependent variable is "
                        + "supported per equation.");
            }
        }
        return name;
    }

    // ------------------------------------------------------------------
    // Error-message rendering
    // ------------------------------------------------------------------

    /**
     * A short, best-effort human-readable label for error messages — a real
     * implementation would reuse ParserNG's existing expression
     * pretty-printer instead of reconstructing one here.
     */
    static String describe(ExprNode node) {
        if (node.kind == ExprNode.Kind.NUMBER) {
            return String.valueOf(node.numberValue);
        }
        if (node.kind == ExprNode.Kind.VARIABLE) {
            return node.isStateVariable() ? node.variableName + "[" + node.stateIndex + "]" : node.variableName;
        }
        String op = node.funcName != null ? node.funcName : String.valueOf(node.opChar);
        StringBuilder sb = new StringBuilder(op).append('(');
        for (int i = 0; i < node.children.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(describe(node.children.get(i)));
        }
        return sb.append(')').toString();
    }
}