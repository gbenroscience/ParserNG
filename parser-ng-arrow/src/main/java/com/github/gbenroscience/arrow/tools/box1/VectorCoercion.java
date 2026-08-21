package com.github.gbenroscience.arrow.tools.box1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;

/**
 * Convenience (NON-zero-copy) conversions from common Arrow numeric
 * vector types into {@link Float8Vector}, for schemas where a column
 * ParserNG needs isn't already float64.
 * <p>
 * Each method here allocates a brand-new {@code Float8Vector} and copies
 * every element — this is a real per-batch cost, not the zero-copy path
 * {@link ArrowBulkEvaluator} otherwise uses. Prefer casting upstream in
 * your own Arrow pipeline (e.g. reading the source as float64 to begin
 * with, or an Arrow compute cast) wherever you control it. Use this class
 * when you don't.
 */
public final class VectorCoercion {

    private VectorCoercion() {
    }

    public static Float8Vector toFloat8(IntVector src, BufferAllocator allocator) {
        int n = src.getValueCount();
        Float8Vector dest = new Float8Vector(src.getField().getName(), allocator);
        dest.allocateNew(n);
        for (int i = 0; i < n; i++) {
            if (src.isNull(i)) {
                dest.setNull(i);
            } else {
                dest.set(i, src.get(i));
            }
        }
        dest.setValueCount(n);
        return dest;
    }

    public static Float8Vector toFloat8(BigIntVector src, BufferAllocator allocator) {
        int n = src.getValueCount();
        Float8Vector dest = new Float8Vector(src.getField().getName(), allocator);
        dest.allocateNew(n);
        for (int i = 0; i < n; i++) {
            if (src.isNull(i)) {
                dest.setNull(i);
            } else {
                dest.set(i, (double) src.get(i));
            }
        }
        dest.setValueCount(n);
        return dest;
    }

    public static Float8Vector toFloat8(Float4Vector src, BufferAllocator allocator) {
        int n = src.getValueCount();
        Float8Vector dest = new Float8Vector(src.getField().getName(), allocator);
        dest.allocateNew(n);
        for (int i = 0; i < n; i++) {
            if (src.isNull(i)) {
                dest.setNull(i);
            } else {
                dest.set(i, (double) src.get(i));
            }
        }
        dest.setValueCount(n);
        return dest;
    }

    /**
     * Dispatches on the runtime type of {@code src}. Throws {@link
     * UnsupportedVectorTypeException} for any vector type not handled
     * above (e.g. {@code DecimalVector}, {@code VarCharVector}) — add a
     * case here following the same pattern if you need one.
     */
    public static Float8Vector toFloat8(FieldVector src, BufferAllocator allocator) {
        if (src instanceof Float8Vector f8) {
            return f8;
        } else if (src instanceof IntVector iv) {
            return toFloat8(iv, allocator);
        } else if (src instanceof BigIntVector bv) {
            return toFloat8(bv, allocator);
        } else if (src instanceof Float4Vector fv) {
            return toFloat8(fv, allocator);
        }
        throw new UnsupportedVectorTypeException(src.getField().getName(), src.getClass());
    }
}
