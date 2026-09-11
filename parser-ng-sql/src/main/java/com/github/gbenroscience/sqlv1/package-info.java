/**
 * jiboyeisrael@gmail.com
 * A SQL-shaped front end for parser-ng-arrow: {@code SELECT ... FROM ...
 * [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] [LIMIT ...]}
 * compiles into vectorized Apache Arrow computations, driven entirely by
 * ParserNG expressions.
 *
 * <h2>Scope</h2>
 * This package deliberately implements a small, closed grammar --
 * documented in full on {@link com.github.gbenroscience.sqlv1.SqlParser} --
 * and nothing more: no {@code JOIN}, no {@code INSERT}/{@code UPDATE}/
 * {@code DELETE}, no transactions, no catalogs, no subqueries. Within that
 * {@code SELECT} shape, {@code CASE}/{@code WHEN}/{@code THEN}/{@code ELSE}/
 * {@code END}, {@code CAST}, {@code GROUP BY}, {@code HAVING},
 * {@code ORDER BY}, and {@code LIMIT} are all supported -- see
 * {@code SqlParser} for how each compiles to ParserNG text, and
 * {@link com.github.gbenroscience.sqlv1.ArrowQuery} ("Aggregation
 * strategy" and "{@code ORDER BY} and {@code LIMIT}") for how grouping,
 * aggregate functions, and sorting are actually executed. The goal, in the
 * words of the original design brief, is
 * <blockquote>"Make SQL a convenient way to describe vectorized Arrow
 * computations." Not: "Make parser-ng-arrow a SQL database."</blockquote>
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
 * <li>The {@code WHERE} clause's boolean structure ({@code AND}/{@code OR}/
 * {@code NOT}/{@code BETWEEN}/{@code IN}/{@code IS [NOT] NULL}/comparisons)
 * is normalized to negation-normal form by
 * {@link com.github.gbenroscience.sqlv1.ast.BoolExprs#toNnf} -- ParserNG has
 * no logical-not operator, so every {@code NOT} is eliminated by pushing it
 * down to the leaves via De Morgan's laws instead.</li>
 * <li>{@link com.github.gbenroscience.sqlv1.ArrowQuery} compiles the
 * resulting expressions against a concrete {@code VectorSchemaRoot}'s
 * schema (lazily, on first {@code execute}, and cached thereafter) and
 * drives parser-ng-arrow's {@code ArrowExpressionEvaluator}s to filter and
 * project.</li>
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