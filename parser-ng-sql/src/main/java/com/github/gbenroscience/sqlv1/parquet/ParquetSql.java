package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SQL over Parquet files.
 *
 * <pre>{@code
 * ParquetSql sql = ParquetSql.create(ParquetSqlOptions.defaults().withParallelism(4))
 *         .register("events", Path.of("data/events.parquet"));
 * try (ParquetSqlPlan p = sql.plan("SELECT id, x * 2 AS x2 FROM events WHERE x > 100")) {
 *     p.stream(alloc, batch -> consume(batch));
 *     System.out.println(p.lastMetrics());
 * }
 * try (VectorSchemaRoot r = sql.execute("SELECT COUNT(*) FROM 'data/events.parquet'", alloc)) { ... }
 * }</pre>
 *
 * <h2>How {@code FROM} is resolved</h2>
 * <ol>
 * <li>{@code FROM 'path/or/glob'} and {@code FROM read_parquet('a', 'b*.parquet')}:
 * the quoted text is the source. Always resolved.</li>
 * <li>{@code FROM name}: looked up among {@link #register registered} tables.
 * If not registered: an error, <em>unless</em> the plan was created with an
 * explicit file ({@link #plan(Path, String)}), in which case the name is
 * ignored exactly as before.</li>
 * <li>An explicit file combined with a quoted path in the SQL is ambiguous and rejected.</li>
 * </ol>
 * The in-memory {@code ArrowSql} path never resolves the name.
 * <b>Quoted paths are file reads:</b> see {@link ParquetTables} and
 * {@link ParquetSqlOptions#withAllowedRoot(Path)}.
 *
 * <p>{@code EXPLAIN SELECT ...} returns the plan as a one-column ({@code plan}) result.
 */
public final class ParquetSql {

    private static final Pattern EXPLAIN = Pattern.compile("^\\s*EXPLAIN\\s+", Pattern.CASE_INSENSITIVE);

    private final ParquetSqlOptions options;
    private final Map<String, String> tables = new ConcurrentHashMap<>();

    private ParquetSql(ParquetSqlOptions options) {
        this.options = options;
    }

    /** @return an instance with default options */
    public static ParquetSql create() {
        return new ParquetSql(ParquetSqlOptions.defaults());
    }

    /** @return an instance with the given options */
    public static ParquetSql create(ParquetSqlOptions options) {
        return new ParquetSql(options);
    }

    /** Registers {@code name} for {@code FROM name}. {@code fileOrGlob} may be a glob. Names are case-sensitive. */
    public ParquetSql register(String name, Path fileOrGlob) {
        return register(name, fileOrGlob.toString());
    }

    /** As {@link #register(String, Path)} with a glob string. */
    public ParquetSql register(String name, String fileOrGlob) {
        if (name == null || name.isBlank()) throw new IllegalArgumentException("table name must not be blank");
        ParquetTables.resolve(fileOrGlob, options.allowedRoot()); // fail at registration, not at first query
        tables.put(name, fileOrGlob);
        return this;
    }

    /** Plans {@code sql}, resolving {@code FROM} per the class comment. */
    public ParquetSqlPlan plan(String sql) {
        return build0(sql, null, options, this);
    }

    /** Plans {@code sql} over one explicit file with this instance's options; a bare {@code FROM name} is ignored. */
    public ParquetSqlPlan planFile(Path file, String sql) {
        return build0(sql, file, options, this);
    }

    /** One-shot execution. Caller closes the result. */
    public VectorSchemaRoot execute(String sql, BufferAllocator allocator) {
        return run(plan(sql), allocator);
    }

    /** One-shot execution over one explicit file with this instance's options. */
    public VectorSchemaRoot executeFile(Path file, String sql, BufferAllocator allocator) {
        return run(planFile(file, sql), allocator);
    }

    // ---- static conveniences: the signatures of the first bridge, unchanged ----

    /** Plans {@code sql} over one file with default options; a bare {@code FROM name} is ignored. */
    public static ParquetSqlPlan plan(Path file, String sql) {
        return build0(sql, file, ParquetSqlOptions.defaults(), null);
    }

    /** As {@link #plan(Path, String)} with explicit options. */
    public static ParquetSqlPlan plan(Path file, String sql, ParquetSqlOptions options) {
        return build0(sql, file, options, null);
    }

    /** One-shot global execution over one file. Caller closes the result. */
    public static VectorSchemaRoot execute(Path file, String sql, BufferAllocator allocator) {
        return run(plan(file, sql), allocator);
    }

    private static VectorSchemaRoot run(ParquetSqlPlan p, BufferAllocator allocator) {
        try (p) {
            if (p.isExplain()) return explainRoot(p.explain(), allocator);
            return p.execute(allocator);
        }
    }

    private static VectorSchemaRoot explainRoot(String text, BufferAllocator allocator) {
        String[] lines = text.split("\n");
        VarCharVector v = new VarCharVector("plan", allocator);
        v.allocateNew(Math.max(64, text.length()), lines.length);
        for (int i = 0; i < lines.length; i++) v.setSafe(i, lines[i].getBytes(StandardCharsets.UTF_8));
        v.setValueCount(lines.length);
        return new VectorSchemaRoot(List.of(v));
    }

    private static ParquetSqlPlan build0(String sql, Path explicit, ParquetSqlOptions o, ParquetSql registry) {
        Matcher m = EXPLAIN.matcher(sql);
        boolean explain = m.find();
        String body = explain ? sql.substring(m.end()) : sql;
        SelectStatement st = SqlParser.parse(body);
        List<Path> files;
        switch (st.tableKind()) {
            case FILE, READ_PARQUET -> {
                if (explicit != null) {
                    throw new ParquetSqlException("FROM names a path but a file was also passed explicitly; use one or the other");
                }
                files = new java.util.ArrayList<>();
                for (String spec : st.tablePaths()) files.addAll(ParquetTables.resolve(spec, o.allowedRoot()));
            }
            case NAME -> {
                if (explicit != null) {
                    files = ParquetTables.resolve(explicit.toString(), o.allowedRoot());
                } else {
                    String spec = registry == null ? null : registry.tables.get(st.table());
                    if (spec == null) {
                        throw new ParquetSqlException("Unknown table '" + st.table() + "'. Register it with register(name, path), "
                                + "or write FROM 'path/to/file.parquet' or FROM read_parquet('glob').");
                    }
                    files = ParquetTables.resolve(spec, o.allowedRoot());
                }
            }
            default -> throw new IllegalStateException();
        }
        return new ParquetSqlPlan(body, st, files, o, explain);
    }
}
