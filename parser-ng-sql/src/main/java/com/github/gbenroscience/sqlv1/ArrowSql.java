package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;

import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * One-shot convenience entry points for parser-ng-sql, for callers who just
 * want to run a query once rather than manage a reusable {@link ArrowQuery}.
 *
 * <pre>{@code
 * VectorSchemaRoot result = ArrowSql.execute(
 *         root, "SELECT sqrt(x*x + y*y) AS distance FROM data WHERE x > 10");
 * }</pre>
 *
 * <p>
 * Every method here parses, compiles, executes exactly once, and closes the
 * underlying {@link ArrowQuery} before returning — appropriate when a given SQL
 * string will only ever run once. For a query that will be executed repeatedly
 * (e.g. once per incoming batch), compile it once via
 * {@link ArrowQuery#compile(String)} and reuse that instance instead;
 * re-parsing and re-compiling the same SQL text on every batch throws away
 * exactly the "parse once, compile once, execute many times" benefit this
 * module exists to provide.
 *
 * @author GBEMIRO
 */
public final class ArrowSql {

    private ArrowSql() {
    }

    /**
     * Compiles {@code sql} and executes it once against {@code root}, using the
     * default backend ({@link ArrowExecutionBackend#CPU_SIMD}) and null policy
     * ({@link NullPolicy#PROPAGATE} — see {@link ArrowQuery}'s "A note on
     * {@code NullPolicy}" for why {@link NullPolicy#IGNORE} is not the default
     * despite its name).
     *
     * @throws SqlSyntaxException if {@code sql} is not well-formed
     * @throws ArrowSqlException if compiling {@code sql}'s expressions against
     * {@code root}'s schema fails
     *
     * @param root
     * @param sql
     * @return
     */
    public static VectorSchemaRoot execute(VectorSchemaRoot root, String sql) {
        try (ArrowQuery query = ArrowQuery.compile(sql)) {
            return query.execute(root);
        }
    }

    /**
     * As {@link #execute(VectorSchemaRoot, String)}, targeting a specific
     * execution backend.
     *
     * @param root
     * @param sql
     * @param backend
     * @return
     */
    public static VectorSchemaRoot execute(VectorSchemaRoot root, String sql, ArrowExecutionBackend backend) {
        try (ArrowQuery query = ArrowQuery.compile(sql).withBackend(backend)) {
            return query.execute(root);
        }
    }

    /**
     * As {@link #execute(VectorSchemaRoot, String)}, targeting a specific
     * execution backend and null policy.
     *
     * @param root
     * @param sql
     * @param backend
     * @param nullPolicy
     * @return
     */
    public static VectorSchemaRoot execute(
            VectorSchemaRoot root, String sql, ArrowExecutionBackend backend, NullPolicy nullPolicy) {
        try (ArrowQuery query = ArrowQuery.compile(sql).withBackend(backend).withNullPolicy(nullPolicy)) {
            return query.execute(root);
        }
    }
}
