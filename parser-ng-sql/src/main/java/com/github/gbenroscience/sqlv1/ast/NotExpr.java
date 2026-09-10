package com.github.gbenroscience.sqlv1.ast;

/**
 * SQL {@code NOT}. Only ever present in a freshly parsed tree — see
 * {@link BoolExpr}'s javadoc and {@link BoolExprs#toNnf(BoolExpr)}, which
 * eliminates it.
 *
 * @author GBEMIRO
 */
public record NotExpr(BoolExpr inner) implements BoolExpr {
}
