package com.github.gbenroscience.sqlv1.ast;

/**
 * The aggregate functions recognized at the top level of a {@code SELECT}
 * item when the query is a {@code GROUP BY}/aggregate query (see
 * {@link SelectItem#isAggregate()} and {@link SelectStatement#isAggregateQuery()}).
 *
 * <p>
 * Each maps to a client-side (Java) reduction over one column of values
 * produced by evaluating the aggregate's argument expression through a
 * compiled ParserNG evaluator, per group — ParserNG itself has no notion of
 * an aggregate function; see {@code ArrowQuery}'s "Aggregation strategy".
 * {@code NULL} values produced by the argument expression are skipped by
 * every aggregate except {@link #COUNT} of {@code *}, matching standard SQL
 * aggregate semantics; an aggregate over a group with no non-null values
 * (including an empty group) produces {@code NULL}, except {@link #COUNT}
 * which produces {@code 0}.
 *
 * @author GBEMIRO
 */
public enum AggregateKind {
    COUNT,
    SUM,
    AVG,
    MIN,
    MAX
}