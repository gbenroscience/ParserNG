package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * The other half of {@link Predicate.Custom}: an exact, row-level test a caller registers by id via
 * {@code ParquetScan.withCustomPredicate(id, impl)}. Invoked only under {@code exactFilter()}, once
 * per row that survived pruning, by {@code internal.RowFilterEvaluator}.
 *
 * <p>This is the intended extension point for anything {@link Predicate}'s own node types have no
 * primitive for — computed/functional expressions ({@code UPPER(x) = 'A'}, {@code x + 1 > 10}), a
 * cross-field business rule, anything. {@code parser-ng-parquet} deliberately does not know or care
 * what the implementation computes; wiring a real expression evaluator (e.g. one already built on
 * {@code parser-ng-arrow}'s {@code ArrowQuery}) behind this interface is the caller's job, kept
 * entirely out of this module so it never needs to depend on one.
 *
 * <h2>Responsibilities this interface does NOT take care of for you</h2>
 * <ul>
 *   <li><b>Null handling</b> is entirely the implementation's own responsibility — unlike every
 *       built-in {@link Predicate} leaf, which excludes a row on a null operand automatically (see
 *       {@link Predicate}'s Javadoc), a {@code CustomPredicate} must check {@code batch.getVector
 *       (name).isNull(row)} itself wherever that matters to its own logic.</li>
 *   <li><b>Thrown exceptions propagate</b>, uncaught, out of the scan; they are not translated into
 *       "row excluded." A custom implementation that can fail should decide for itself whether a
 *       failure means "exclude this row" (return {@code false}) or "this scan cannot continue"
 *       (throw), rather than have that choice made for it.</li>
 * </ul>
 */
@FunctionalInterface
public interface CustomPredicate {

    /**
     * @param batch the current decoded batch (never null); read via {@code batch.getVector(name)} —
     *              the columns named in the corresponding {@link Predicate.Custom#touchedColumns()}
     *              are guaranteed present (validated before any row is scanned), but nothing stops an
     *              implementation from reading others too if it has a reason to
     * @param row   row index within {@code batch}
     * @return true if this row should be kept
     */
    boolean test(VectorSchemaRoot batch, int row);
}
