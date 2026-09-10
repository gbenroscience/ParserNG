package com.github.gbenroscience.sqlv1.ast;

/**
 * A node in the {@code WHERE}-clause boolean AST built by
 * {@code SqlParser.parseBooleanExpression}.
 *
 * <p>
 * This directly mirrors the grammar's {@code boolean_expression} /
 * {@code or_expression} / {@code and_expression} / {@code not_expression} /
 * {@code predicate} productions:
 * <ul>
 * <li>{@link AndExpr}, {@link OrExpr} — {@code AND} / {@code OR}</li>
 * <li>{@link NotExpr} — {@code NOT}; only ever appears in the
 * <i>freshly parsed</i> tree. {@link BoolExprs#toNnf(BoolExpr)} eliminates
 * every {@link NotExpr} by pushing negation down to the leaves (ParserNG has
 * no logical-not operator to compile {@code NOT} against directly — see that
 * method's javadoc), so a tree that has been through {@code toNnf} never
 * contains one.</li>
 * <li>{@link ComparisonExpr}, {@link BetweenExpr}, {@link InExpr},
 * {@link IsNullExpr} — the four {@code predicate} leaf forms. The fifth
 * grammar alternative, {@code '(' boolean_expression ')'}, has no AST node
 * of its own: parenthesization is structural only (it controls how the
 * parser nests these very node types) and carries no meaning once
 * parsing is done.</li>
 * </ul>
 *
 * <p>
 * Every operand string held by a leaf node (the two sides of a
 * {@link ComparisonExpr}, a {@link BetweenExpr}'s bounds, an
 * {@link InExpr}'s values, the target of an {@link IsNullExpr}) is raw
 * ParserNG expression source text, captured verbatim from the original SQL —
 * see {@code SqlParser}'s "Why expressions are captured, not rebuilt".
 *
 * @author GBEMIRO
 */
public sealed interface BoolExpr
        permits AndExpr, OrExpr, NotExpr, ComparisonExpr, BetweenExpr, InExpr, IsNullExpr {
}
