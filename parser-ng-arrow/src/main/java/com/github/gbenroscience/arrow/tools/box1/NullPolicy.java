package com.github.gbenroscience.arrow.tools.box1;

/**
 * Governs how {@link ArrowBulkEvaluator} handles Arrow null (validity
 * bitmap) entries in bound input columns.
 * <p>
 * SIMDEngineEvaluator's kernels have no concept of null — they operate on
 * dense double lanes. Arrow's null semantics have to be reconciled with
 * that at the boundary, and there is no policy-free default: silently
 * computing over an Arrow "null" slot's underlying data lane will not
 * crash, but the value it computes with is whatever bytes happen to sit
 * in that buffer, which the Arrow spec explicitly leaves unspecified for
 * null slots. That's why this is a required, explicit choice.
 */
public enum NullPolicy {

    /**
     * Default. Before evaluating, scans every bound input column's
     * validity bitmap. If any row in {@code [0, rowCount)} is null in any
     * bound column, throws {@link ArrowNullValueException} instead of
     * computing over unspecified data. Use this when your pipeline should
     * fail loudly on unexpected nulls.
     */
    REJECT_ON_NULL,

    /**
     * Computes over every row's underlying data lane unconditionally —
     * this preserves the zero-copy fast path, since no per-element branch
     * is needed — then overwrites the OUTPUT vector's validity bitmap
     * with the bitwise AND of every bound input column's validity bitmap
     * (a result row is non-null only if it was non-null in every input
     * this expression reads). The AND is a block byte-wise operation over
     * {@link java.lang.foreign.MemorySegment}, not a per-row branch, so it
     * stays cheap relative to the arithmetic itself.
     * <p>
     * The output DATA lane for a null result row is whatever the kernel
     * computed from the input's unspecified data — it is not guaranteed
     * to be {@code 0.0} or {@code NaN}. Consumers must respect the
     * validity bitmap, exactly as for any other Arrow-produced null.
     */
    PROPAGATE_NULL
}
