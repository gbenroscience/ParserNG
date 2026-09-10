package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code select_item ::= expression [AS identifier]}.
 *
 * @param exprText raw ParserNG expression text, captured verbatim from the
 * source SQL
 * @param alias the {@code AS} name, or {@code null} if none was given — an
 * unaliased column is named after its own (trimmed) {@code exprText}, the
 * same convention {@code ArrowExpressionEvaluators.filterProject} uses in
 * parser-ng-arrow
 *
 * @author GBEMIRO
 */
public record SelectItem(String exprText, String alias) {

    /**
     * @return {@link #alias()} if present, otherwise {@link #exprText()} —
     * the name this column will have in the result batch
     */
    public String outputName() {
        return alias != null ? alias : exprText;
    }
}
