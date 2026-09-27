package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

/** Internal: a stream of Arrow batches. Sequential and parallel scans implement it identically. */
interface BatchSource extends AutoCloseable {

    Schema schema();

    /** Advances; false at end of data. */
    boolean next();

    /** The current batch; valid until the next {@link #next()} or {@link #close()}. */
    VectorSchemaRoot root();

    /** Hands the current batch to the caller (zero-copy). */
    VectorSchemaRoot detach();

    @Override
    void close();
}
