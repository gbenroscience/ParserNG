package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code select_item ::= expression [AS identifier] | aggregate_call [AS identifier]}.
 *
 * <p>
 * A plain (non-aggregate) item is built with {@link #SelectItem(String, String)}
 * exactly as before: {@code exprText} is raw ParserNG expression text
 * (which may itself contain a {@code CASE}/{@code CAST} construct rendered
 * to ParserNG text by {@code SqlParser} — see its javadoc), {@link #aggregateKind()}
 * is {@code null}, and {@link #isAggregate()} is {@code false}.
 *
 * <p>
 * An aggregate item ({@code COUNT}/{@code SUM}/{@code AVG}/{@code MIN}/
 * {@code MAX}, only recognized at the top level of a select item — see
 * {@code SqlParser}) is built with {@link #aggregate(AggregateKind, boolean,
 * String, String, String)}: {@link #exprText()} holds the canonical,
 * reconstructed call text (e.g. {@code "SUM(x * y)"} or {@code "COUNT(*)"}),
 * used both as the default (unaliased) output column name and — when it
 * happens to be referenced verbatim in a {@code HAVING}/{@code ORDER BY}
 * clause — as the name {@code ArrowQuery} materializes the aggregated
 * result column under, so the two resolve to the same column. See
 * {@code ArrowQuery}'s "Aggregation strategy" for why aliasing an aggregate
 * item is recommended when referencing it from {@code HAVING}/{@code ORDER BY}.
 *
 * @param exprText raw ParserNG expression text for a plain item, or the
 * canonical reconstructed call text ({@code "FUNC(arg)"}/{@code "COUNT(*)"})
 * for an aggregate item
 * @param alias the {@code AS} name, or {@code null} if none was given — an
 * unaliased column is named after its own (trimmed) {@code exprText}, the
 * same convention {@code ArrowExpressionEvaluators.filterProject} uses in
 * parser-ng-arrow
 * @param aggregateKind {@code null} for a plain item; the recognized
 * aggregate function for an aggregate item
 * @param aggregateStar {@code true} only for {@code COUNT(*)}
 * @param aggregateArgText raw ParserNG expression text for the aggregate's
 * single argument, or {@code null} for {@code COUNT(*)} or a plain item
 *
 * @author GBEMIRO
 */
public record SelectItem(
        String exprText, String alias, AggregateKind aggregateKind, boolean aggregateStar, String aggregateArgText) {

    /**
     * Builds a plain (non-aggregate) select item, exactly as parser-ng-sql
     * v1 always has.
     */
    public SelectItem(String exprText, String alias) {
        this(exprText, alias, null, false, null);
    }

    /**
     * Builds an aggregate select item.
     *
     * @param kind the recognized aggregate function
     * @param star {@code true} for {@code COUNT(*)}
     * @param argText raw ParserNG expression text for the argument, or
     * {@code null} iff {@code star} is {@code true}
     * @param canonicalText the reconstructed {@code "FUNC(arg)"}/
     * {@code "COUNT(*)"} text, used as {@link #exprText()}
     * @param alias the {@code AS} name, or {@code null}
     */
    public static SelectItem aggregate(
            AggregateKind kind, boolean star, String argText, String canonicalText, String alias) {
        if (kind == null) {
            throw new NullPointerException("kind must not be null");
        }
        if (star && argText != null) {
            throw new IllegalArgumentException("COUNT(*) must not also carry an argument expression");
        }
        if (!star && argText == null) {
            throw new IllegalArgumentException("A non-star aggregate requires an argument expression");
        }
        return new SelectItem(canonicalText, alias, kind, star, argText);
    }

    /**
     * @return {@code true} iff this is an aggregate item ({@link #aggregateKind()}
     * is non-{@code null})
     */
    public boolean isAggregate() {
        return aggregateKind != null;
    }

    /**
     * @return {@link #alias()} if present, otherwise {@link #exprText()} —
     * the name this column will have in the result batch
     */
    public String outputName() {
        return alias != null ? alias : exprText;
    }
}