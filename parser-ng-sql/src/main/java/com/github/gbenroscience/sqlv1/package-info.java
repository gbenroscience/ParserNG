/**
 * jiboyeisrael@gmail.com
 * A SQL-shaped front end for parser-ng-arrow: {@code SELECT ... FROM ...
 * [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] [LIMIT ...]}
 * compiles into vectorized Apache Arrow computations, driven by ParserNG
 * expressions wherever possible.
 *
 * <h2>Scope</h2>
 * This package deliberately implements a closed grammar -- documented in
 * full on {@link com.github.gbenroscience.sqlv1.SqlParser} -- and nothing
 * more: no {@code JOIN}, no {@code INSERT}/{@code UPDATE}/{@code DELETE},
 * no transactions, no catalogs, no subqueries. 
 * <ul>
 * <li><b>Embedded boolean conditions</b> -- a comparison/{@code AND}/
 * {@code OR}/{@code NOT}/{@code BETWEEN}/{@code IN} usable as a function
 * argument, a grouping-paren term, or a {@code select_item}'s own top-level
 * expression (e.g. {@code if(sin(x) > 0, tan(x), 0.2)}), not just inside
 * {@code WHERE}/{@code HAVING}. See {@code SqlParser}'s "Embedded boolean
 * conditions".</li>
 * <li><b>{@code GROUP BY}/{@code HAVING}/{@code ORDER BY}/{@code LIMIT},
 * aggregates ({@code SUM}/{@code COUNT}/{@code AVG}/{@code MIN}/
 * {@code MAX}), and {@code CASE}/{@code CAST}</b> -- a deliberate,
 * explicitly-approved expansion past the original v1 shape, still stopping
 * well short of a general query engine (no window functions, no nested/
 * correlated aggregation, no {@code JOIN} to aggregate across). See
 * {@code SqlParser}'s "{@code GROUP BY} / aggregates" and "{@code CASE} /
 * {@code CAST}", and {@code SelectStatement}'s "{@code GROUP BY} /
 * aggregates -- a deliberately strict subset" for exactly which
 * non-aggregate {@code SELECT} items a grouped query may contain.</li>
 * </ul>
 *
 * <h2>How a query gets from text to a compiled kernel</h2>
 * <ol>
 * <li>{@link com.github.gbenroscience.sqlv1.SqlLexer} tokenizes the SQL
 * text.</li>
 * <li>{@link com.github.gbenroscience.sqlv1.SqlParser} parses it into a
 * {@link com.github.gbenroscience.sqlv1.ast.SelectStatement}. Arithmetic
 * {@code expression}s are never rebuilt into a tree -- their exact source
 * text is captured and handed straight to ParserNG's own compiler
 * downstream, so every inbuilt (and user-registered) ParserNG function is
 * supported automatically. See {@code SqlParser}'s "Why expressions are
 * captured, not rebuilt".</li>
 * <li>The {@code WHERE}/{@code HAVING} clauses' boolean structure
 * ({@code AND}/{@code OR}/{@code NOT}/{@code BETWEEN}/{@code IN}/
 * {@code IS [NOT] NULL}/comparisons) is normalized to negation-normal form
 * by {@link com.github.gbenroscience.sqlv1.ast.BoolExprs#toNnf} -- ParserNG
 * has no logical-not operator, so every {@code NOT} is eliminated by
 * pushing it down to the leaves via De Morgan's laws instead. {@code CASE}
 * conditions go through the same machinery.</li>
 * <li>{@link com.github.gbenroscience.sqlv1.ArrowQuery} compiles the
 * resulting expressions against a concrete {@code VectorSchemaRoot}'s
 * schema (lazily, on first {@code execute}, and cached thereafter) and
 * drives parser-ng-arrow's {@code ArrowExpressionEvaluator}s to filter and
 * project. For a grouped query, {@code ArrowQuery} also evaluates
 * {@code GROUP BY}/aggregation/{@code HAVING} itself, in Java, after
 * ParserNG has computed every key and aggregate argument -- ParserNG has no
 * concept of grouping. {@code ORDER BY}/{@code LIMIT} are applied last, at
 * whichever pipeline stage matches the query's shape (grouped vs.
 * non-grouped) -- see {@code ArrowQuery}'s own "{@code GROUP BY} /
 * {@code HAVING} / {@code ORDER BY} / {@code LIMIT}" section.</li>
 * </ol>
 *
 * <h2>Where to start</h2>
 * {@link com.github.gbenroscience.sqlv1.ArrowSql} for one-off queries;
 * {@link com.github.gbenroscience.sqlv1.ArrowQuery} for anything executed
 * more than once.
 *
 * @author GBEMIRO
 */
package com.github.gbenroscience.sqlv1;