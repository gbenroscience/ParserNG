package com.github.gbenroscience.sqlv1.ast;

/**
 * SQL {@code AND}. Renders to ParserNG's {@code &&}.
 *
 * @author GBEMIRO
 */
public record AndExpr(BoolExpr left, BoolExpr right) implements BoolExpr {
}
