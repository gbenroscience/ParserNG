package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code select_item ::= (expression | aggregate_call) [AS identifier]}.
 *
 * <p>
 * Two shapes are represented here, distinguished by {@link #aggregate()}:
 * <ul>
 * <li><b>Plain</b> ({@link #aggregate()} is {@code null}): {@link #exprText()}
 * is raw ParserNG expression text, captured verbatim from the source SQL,
 * exactly as before this field was added.
 * <li><b>Aggregate</b> ({@link #aggregate()} is non-{@code null}, e.g.
 * {@code SUM(x*y)}): {@link #exprText()} is a synthetic description
 * ({@link AggregateSpec#defaultOutputName()}) used only as the default
 * output name when no {@code AS} alias is given — it is never itself
 * compiled by ParserNG; {@code ArrowQuery} dispatches on
 * {@link #aggregate()} instead. See {@link AggFunc}'s javadoc for why
 * aggregation is handled entirely outside ParserNG.
 * </ul>
 * Construct via {@link #plain(String, String)} or
 * {@link #aggregate(AggregateSpec, String)} rather than the canonical
 * constructor directly, so the {@link #exprText()} synthesis for the
 * aggregate case can never drift from {@link AggregateSpec#defaultOutputName()}.
 *
 * @param exprText see above
 * @param alias the {@code AS} name, or {@code null} if none was given — an
 * unaliased column is named after its own (trimmed) {@code exprText}, the
 * same convention {@code ArrowExpressionEvaluators.filterProject} uses in
 * parser-ng-arrow
 * @param aggregate non-{@code null} iff this item is an aggregate call
 *
 * @author GBEMIRO
 */
public record SelectItem(String exprText, String alias, AggregateSpec aggregate) {

    /**
     * @return a plain (non-aggregate) select item
     */
    public static SelectItem plain(String exprText, String alias) {
        return new SelectItem(exprText, alias, null);
    }

    /**
     * @return an aggregate select item, e.g. {@code SUM(x*y) AS total}
     */
    public static SelectItem aggregate(AggregateSpec spec, String alias) {
        return new SelectItem(spec.defaultOutputName(), alias, spec);
    }

    /**
     * @return {@code true} iff this item is an aggregate call ({@link #aggregate()}
     * is non-{@code null})
     */
    public boolean isAggregate() {
        return aggregate != null;
    }

    /**
     * @return {@link #alias()} if present, otherwise {@link #exprText()} —
     * the name this column will have in the result batch
     */
    public String outputName() {
        return alias != null ? alias : exprText;
    }
}
