package com.github.gbenroscience.sqlv1.ast;

/**
 * SQL {@code OR}. Renders to ParserNG's {@code ||}.
 *
 * @author GBEMIRO
 */
public record OrExpr(BoolExpr left, BoolExpr right) implements BoolExpr {
}
