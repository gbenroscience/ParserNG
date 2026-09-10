package com.github.gbenroscience.sqlv1.ast;

/**
 * A SQL {@code comparison_operator}, together with its ParserNG rendering and
 * its logical negation (for pushing {@code NOT} down to negation-normal
 * form — see {@link BoolExprs#toNnf(BoolExpr)}).
 *
 * <p>
 * SQL's {@code '='} maps to ParserNG's {@code "=="}; SQL's {@code '<>'} and
 * {@code '!='} both map to ParserNG's {@code "!="}; the ordering operators
 * are spelled identically in both languages.
 *
 * @author GBEMIRO
 */
public enum CompOp {
    EQ("=="),
    NEQ("!="),
    LT("<"),
    LE("<="),
    GT(">"),
    GE(">=");

    private final String parserNgSymbol;

    CompOp(String parserNgSymbol) {
        this.parserNgSymbol = parserNgSymbol;
    }

    /**
     * @return this operator's spelling in ParserNG's own expression syntax
     */
    public String parserNgSymbol() {
        return parserNgSymbol;
    }

    /**
     * @return the operator such that {@code NOT (a THIS b)} is equivalent to
     * {@code a negate() b}, e.g. {@code negate(LT) == GE}
     */
    public CompOp negate() {
        return switch (this) {
            case EQ -> NEQ;
            case NEQ -> EQ;
            case LT -> GE;
            case LE -> GT;
            case GT -> LE;
            case GE -> LT;
        };
    }
}
