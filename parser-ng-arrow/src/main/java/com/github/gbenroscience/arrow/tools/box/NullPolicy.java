package com.github.gbenroscience.arrow.tools.box;

/**
 * Controls how {@link ArrowBulkEvaluator} handles Arrow validity (null)
 * bitmaps during evaluation.
 *
 * <p>Bulk evaluation operates directly on each column's raw data buffer via
 * {@link ArrowMemoryBridge} — it does not read validity bitmaps as part of
 * the numeric computation itself, since doing so would require a per-element
 * branch that defeats the point of the SIMD fast paths. This policy governs
 * only whether a separate, cheap pass over the (much smaller) validity
 * bitmaps runs afterward.
 */
public enum NullPolicy {

    /**
     * Fastest option, and the default. The evaluator does not inspect Arrow
     * validity bitmaps at all. Rows where a bound input column is null will
     * still be evaluated using whatever bit pattern happens to occupy that
     * column's data buffer at that position — Arrow does not guarantee this
     * is any particular value (it is commonly, but not reliably, left at 0
     * or a prior value) — and the output's own validity bitmap is left
     * exactly as the caller supplied it. Use this when the caller can
     * guarantee the bound columns have no nulls, or when null handling will
     * be performed separately by the caller.
     */
    IGNORE,

    /**
     * The output row is marked null (via the output vector's validity
     * bitmap) if ANY bound input column is null at that row — standard
     * SQL/Arrow-style null propagation. The corresponding output data value
     * at that row is left as whatever the arithmetic happened to produce;
     * only the validity bit is authoritative once this policy is used, and
     * callers must not read the data value at a null row without checking
     * validity first. This adds one bitwise-AND pass over the validity
     * bitmaps per bound column — cheap relative to the data evaluation
     * itself, since validity bitmaps are 64x smaller than the corresponding
     * double data buffers (1 bit per row vs. 64).
     */
    PROPAGATE
}