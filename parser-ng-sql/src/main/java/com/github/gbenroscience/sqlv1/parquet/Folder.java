package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * A query answered in Java over the batches of an <em>inner</em> plain projection (same {@code FROM} and
 * {@code WHERE}, so pruning, native predicates and streaming apply to the scan): {@link FoldQuery} for
 * global aggregates and {@code SELECT DISTINCT}, {@link GroupFold} for {@code GROUP BY}.
 */
interface Folder {

    /** The projection the scan runs. */
    SelectStatement inner();

    /** Folds the inner plan's batches into the result; the caller closes it. */
    VectorSchemaRoot execute(BufferAllocator alloc, ParquetSqlPlan inner, long maxRows);
}
