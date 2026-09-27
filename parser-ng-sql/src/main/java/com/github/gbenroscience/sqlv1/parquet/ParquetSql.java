package com.github.gbenroscience.sqlv1.parquet;
  
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import com.github.gbenroscience.sqlv1.ArrowQuery;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.nio.file.Path;

/**
 * Entry point of the bridge.
 *
 * <pre>{@code
 * try (BufferAllocator alloc = new RootAllocator();
 *      ParquetSqlPlan p = ParquetSql.plan(file,
 *          "SELECT x, y, sqrt(x*x+y*y) AS magnitude FROM t WHERE x > 100 AND sqrt(x*x+y*y) < 500")) {
 *     p.stream(alloc, batch -> consume(batch));          // bounded memory
 * }
 * try (VectorSchemaRoot r = ParquetSql.execute(file, "SELECT ... ORDER BY magnitude LIMIT 20", alloc)) { ... }
 * }</pre>
 *
 * <p>Division of labour: this class reads the file footer (cheap), asks {@link ScanPlanner} which
 * columns and pruning predicate to use, and hands everything else to {@link ArrowQuery} untouched.
 */
public final class ParquetSql {

    private ParquetSql() { }

    public static ParquetSqlPlan plan(Path file, String sql) {
        ParquetFileInfo info = ParquetFileInfo.read(file);
        ArrowQuery q = ArrowQuery.compile(sql);
        try {
            ScanPlanner.ScanPlan sp = ScanPlanner.plan(q.statement(), info.topLevelColumns(), info.primitiveColumns());
            ParquetScan scan = ParquetScan.scan(file).withMetrics(true);
            if (sp.columns() != null) scan = scan.select(sp.columns().toArray(new String[0]));
            if (sp.pruning() != null) scan = scan.pushdown(sp.pruning());
            return new ParquetSqlPlan(q, sp, scan);
        } catch (RuntimeException e) {
            q.close();
            throw e;
        }
    }

    /** One-shot global execution. Caller closes the result. */
    public static VectorSchemaRoot execute(Path file, String sql, BufferAllocator allocator) {
        try (ParquetSqlPlan p = plan(file, sql)) {
            return p.execute(allocator);
        }
    }
}
