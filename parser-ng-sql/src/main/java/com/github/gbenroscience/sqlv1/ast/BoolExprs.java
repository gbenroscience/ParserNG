package com.github.gbenroscience.sqlv1.ast;

/**
 * Transform and render operations over {@link BoolExpr} trees.
 *
 * <h2>Why negation-normal form</h2>
 * ParserNG's own expression grammar has {@code &&}, {@code ||}, and the six
 * comparison operators, but no logical-not operator — its {@code "!"} token
 * is postfix factorial, not boolean negation (see
 * {@code com.github.gbenroscience.parser.OperatorConstant.FACTORIAL}
 * upstream in parser-ng). A SQL {@code NOT} therefore cannot be compiled by
 * simply prepending {@code !} to a rendered sub-expression the way it could
 * in a language that has a native boolean-not.
 *
 * <p>
 * {@link #toNnf(BoolExpr)} sidesteps this entirely by pushing every
 * {@code NOT} down to the leaves via De Morgan's laws until it disappears:
 * {@code NOT (a AND b)} becomes {@code (NOT a) OR (NOT b)},
 * {@code NOT (a OR b)} becomes {@code (NOT a) AND (NOT b)}, and
 * {@code NOT} of a comparison/{@code BETWEEN}/{@code IN}/{@code IS NULL}
 * leaf is resolved by flipping that leaf in place ({@code <} becomes
 * {@code >=}, a {@code BETWEEN}/{@code IN}/{@code IS NULL}'s own
 * {@code negated} flag flips) — see {@link CompOp#negate()},
 * {@link BetweenExpr#negate()}, {@link InExpr#negate()},
 * {@link IsNullExpr#negate()}. The grammar's {@code not_expression} only
 * ever wraps a {@code predicate} (a leaf, or a further {@code NOT}, or a
 * parenthesized {@code boolean_expression}) — never a bare arithmetic
 * {@code expression} — so this rewrite is total: a tree that has been
 * through {@code toNnf} is guaranteed to contain no {@link NotExpr}.
 *
 * <h2>Two rendering strategies</h2>
 * {@link #renderFused(BoolExpr)} renders an entire (NNF) tree as a single
 * ParserNG boolean expression string, for one fused, single-kernel
 * evaluation — but only works when the tree contains no {@link IsNullExpr}
 * anywhere, since that leaf type has no ParserNG rendering at all (see its
 * javadoc). Check {@link #containsIsNull(BoolExpr)} first;
 * {@code ArrowQuery} uses it to decide between the fast single-predicate
 * path and the leaf-by-leaf mask-evaluation fallback that handles
 * {@code IS [NOT] NULL}.
 *
 * @author GBEMIRO
 */
public final class BoolExprs {

    private BoolExprs() {
    }

    /**
     * Rewrites {@code expr} to negation-normal form: every {@link NotExpr}
     * is eliminated by pushing it down to the leaves. See the class javadoc.
     */
    public static BoolExpr toNnf(BoolExpr expr) {
        return switch (expr) {
            case AndExpr(var l, var r) -> new AndExpr(toNnf(l), toNnf(r));
            case OrExpr(var l, var r) -> new OrExpr(toNnf(l), toNnf(r));
            case NotExpr(var inner) -> negate(inner);
            case ComparisonExpr c -> c;
            case BetweenExpr b -> b;
            case InExpr i -> i;
            case IsNullExpr n -> n;
        };
    }

    /**
     * Returns the negation-normal form of {@code NOT expr}, i.e.
     * {@code toNnf(new NotExpr(expr))} without actually allocating the
     * intermediate {@link NotExpr}. Used by {@link #toNnf(BoolExpr)} itself
     * and available directly for callers building trees programmatically.
     */
    public static BoolExpr negate(BoolExpr expr) {
        return switch (expr) {
            case AndExpr(var l, var r) -> new OrExpr(negate(l), negate(r));
            case OrExpr(var l, var r) -> new AndExpr(negate(l), negate(r));
            case NotExpr(var inner) -> toNnf(inner);
            case ComparisonExpr(var l, var op, var r) -> new ComparisonExpr(l, op.negate(), r);
            case BetweenExpr b -> b.negate();
            case InExpr i -> i.negate();
            case IsNullExpr n -> n.negate();
        };
    }

    /**
     * @return {@code true} if {@code expr} contains an {@link IsNullExpr}
     * anywhere in its tree
     */
    public static boolean containsIsNull(BoolExpr expr) {
        return switch (expr) {
            case AndExpr(var l, var r) -> containsIsNull(l) || containsIsNull(r);
            case OrExpr(var l, var r) -> containsIsNull(l) || containsIsNull(r);
            case NotExpr(var inner) -> containsIsNull(inner);
            case ComparisonExpr c -> false;
            case BetweenExpr b -> false;
            case InExpr i -> false;
            case IsNullExpr n -> true;
        };
    }

    /**
     * Renders a single leaf node ({@link ComparisonExpr}, {@link BetweenExpr}
     * or {@link InExpr}) as a standalone, self-parenthesized ParserNG boolean
     * expression fragment — safe to combine with {@code &&}/{@code ||}
     * without worrying about operator precedence against whatever surrounds
     * it.
     *
     * @throws IllegalArgumentException if {@code leaf} is an
     * {@link IsNullExpr} (not renderable — see its javadoc), or an
     * {@link AndExpr}/{@link OrExpr}/{@link NotExpr} (not a leaf; use
     * {@link #renderFused(BoolExpr)} for a full tree)
     */
    public static String renderLeaf(BoolExpr leaf) {
        return switch (leaf) {
            case ComparisonExpr(var l, var op, var r) ->
                "(" + l + " " + op.parserNgSymbol() + " " + r + ")";
            case BetweenExpr(var target, var low, var high, var negated) -> negated
                    ? "(" + target + " < " + low + " || " + target + " > " + high + ")"
                    : "(" + target + " >= " + low + " && " + target + " <= " + high + ")";
            case InExpr(var target, var values, var negated) -> {
                StringBuilder sb = new StringBuilder("(");
                String joiner = negated ? " && " : " || ";
                String eq = negated ? " != " : " == ";
                for (int i = 0; i < values.size(); i++) {
                    if (i > 0) {
                        sb.append(joiner);
                    }
                    sb.append(target).append(eq).append(values.get(i));
                }
                yield sb.append(")").toString();
            }
            case IsNullExpr n -> throw new IllegalArgumentException(
                    "IS [NOT] NULL has no ParserNG rendering; ArrowQuery must evaluate "
                    + "IsNullExpr leaves directly against Arrow validity bitmaps instead "
                    + "of calling renderLeaf/renderFused on them.");
            case AndExpr a -> throw new IllegalArgumentException("renderLeaf called on an AndExpr; not a leaf");
            case OrExpr o -> throw new IllegalArgumentException("renderLeaf called on an OrExpr; not a leaf");
            case NotExpr n -> throw new IllegalArgumentException(
                    "renderLeaf called on a NotExpr; run toNnf(...) first");
        };
    }

    /**
     * Renders an entire tree (which must already be in NNF, and must
     * contain no {@link IsNullExpr} — check {@link #containsIsNull(BoolExpr)}
     * first) as a single ParserNG boolean expression string, suitable for
     * one fused {@code ArrowExpressionEvaluators.compile}/{@code evaluate}
     * call.
     *
     * @throws IllegalArgumentException if {@code expr} (or any
     * sub-expression) is an {@link IsNullExpr} or a {@link NotExpr}
     */
    public static String renderFused(BoolExpr expr) {
        return switch (expr) {
            case AndExpr(var l, var r) -> "(" + renderFused(l) + " && " + renderFused(r) + ")";
            case OrExpr(var l, var r) -> "(" + renderFused(l) + " || " + renderFused(r) + ")";
            case NotExpr n -> throw new IllegalArgumentException(
                    "renderFused called on a tree containing NotExpr; run toNnf(...) first");
            default -> renderLeaf(expr);
        };
    }
}
