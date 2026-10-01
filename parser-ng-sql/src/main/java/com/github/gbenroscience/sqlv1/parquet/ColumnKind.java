package com.github.gbenroscience.sqlv1.parquet;

/**
 * The bridge's view of a Parquet column, derived from the Arrow type that
 * parser-ng-parquet decodes it into. Decides how a column may be used:
 * numeric kinds can enter expressions (widened to {@code double}); every kind
 * can be selected as-is (passthrough, native type preserved).
 */
enum ColumnKind {
    INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
    FLOAT32, FLOAT64, DECIMAL,
    BOOL, STRING, BINARY, DATE, TIMESTAMP,
    /** Nested (struct/list/map), UUID, or anything else: selectable, never usable in expressions. */
    OTHER;

    /** @return true if a column of this kind may appear in a numeric expression or numeric predicate */
    boolean numeric() {
        return switch (this) {
            case INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT32, FLOAT64, DECIMAL -> true;
            default -> false;
        };
    }

    /** @return true for kinds whose widening to double can lose precision beyond 2^53 */
    boolean wide() {
        return this == INT64 || this == UINT64 || this == DECIMAL;
    }
}
