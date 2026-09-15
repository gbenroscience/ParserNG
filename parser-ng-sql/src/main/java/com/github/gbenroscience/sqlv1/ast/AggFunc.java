package com.github.gbenroscience.sqlv1.ast;

import java.util.Locale;

/**
 * The aggregate functions {@code GROUP BY} queries may use in the
 * {@code SELECT} list: {@code SUM}, {@code COUNT}, {@code AVG}, {@code MIN},
 * {@code MAX}. Recognized case-insensitively by {@link #fromName(String)},
 * exactly like every other SQL keyword in this grammar — but these are
 * <em>not</em> lexer keywords (see {@code SqlLexer}); they stay ordinary
 * {@code IDENTIFIER} tokens, disambiguated contextually by
 * {@code SqlParser#tryParseAggregate()} only when immediately followed by
 * {@code '('}, so a real column or ParserNG function that happens to be
 * named e.g. {@code sum} is never shadowed.
 *
 * <p>
 * Every aggregate here operates purely in Java over a per-group accumulator
 * (see {@code ArrowQuery}'s {@code Aggregator} implementations) — ParserNG's
 * own expression evaluators are row-wise/vectorized and have no notion of
 * cross-row aggregation, so this is deliberately outside ParserNG's
 * territory, unlike every other expression construct in this module.
 *
 * @author GBEMIRO
 */
public enum AggFunc {
    SUM,
    COUNT,
    AVG,
    MIN,
    MAX;

    /**
     * @return the {@link AggFunc} whose name matches {@code text}
     * case-insensitively, or {@code null} if none does (meaning: this is an
     * ordinary identifier/function name, not an aggregate call)
     */
    public static AggFunc fromName(String text) {
        for (AggFunc f : values()) {
            if (f.name().equalsIgnoreCase(text)) {
                return f;
            }
        }
        return null;
    }

    /**
     * @return this function's canonical SQL spelling, e.g. for use in a
     * synthetic default output-column name ({@code "SUM"}, not
     * {@code "sum"} — matches how the source SQL is conventionally cased,
     * though SQL itself is case-insensitive on this keyword)
     */
    public String sqlName() {
        return name().toUpperCase(Locale.ROOT);
    }
}
