package com.github.gbenroscience.sqlv1.ast;

/**
 * One aggregate function call in a {@code SELECT} list, e.g. {@code SUM(x*y)}
 * or {@code COUNT(*)}.
 *
 * @param func which aggregate function
 * @param argExprText raw ParserNG expression text for the argument (e.g.
 * {@code "x * y"} for {@code SUM(x*y)}) — {@code null} iff {@link #star()}
 * is {@code true}, since {@code COUNT(*)} has no per-row expression to
 * evaluate at all (every row counts, unconditionally)
 * @param star {@code true} for the {@code COUNT(*)} form specifically;
 * {@code false} for every other aggregate call, including
 * {@code COUNT(expr)}
 *
 * @author GBEMIRO
 */
public record AggregateSpec(AggFunc func, String argExprText, boolean star) {

    public AggregateSpec {
        if (star && func != AggFunc.COUNT) {
            throw new IllegalArgumentException("Only COUNT(*) is valid; " + func + "(*) is not");
        }
        if (star && argExprText != null) {
            throw new IllegalArgumentException("A star aggregate must not also carry an argument expression");
        }
        if (!star && argExprText == null) {
            throw new IllegalArgumentException("A non-star aggregate must carry an argument expression");
        }
    }

    /**
     * @return the default output-column name for this aggregate when the
     * {@code SELECT} item carries no {@code AS} alias, e.g.
     * {@code "SUM(x * y)"} or {@code "COUNT(*)"} — mirrors the convention
     * {@link SelectItem#outputName()} already uses for plain expressions
     * (named after their own trimmed text)
     */
    public String defaultOutputName() {
        return star ? func.sqlName() + "(*)" : func.sqlName() + "(" + argExprText + ")";
    }
}
