package com.github.gbenroscience.arrow.tools.box1;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.Float8Vector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Zero-copy bridge between Arrow's off-heap {@link ArrowBuf} buffers and
 * the JDK Foreign Function &amp; Memory API's {@link MemorySegment}, which
 * is what SIMDEngineEvaluator's {@code applyBulk(MemorySegment[],
 * MemorySegment)} API consumes.
 * <p>
 * Arrow's {@code ArrowBuf} is backed by real off-heap native memory (via
 * Netty's allocator, in the {@code arrow-memory-netty} implementation).
 * {@link MemorySegment#ofAddress(long)} combined with {@link
 * MemorySegment#reinterpret(long)} constructs a segment view directly
 * over that same native address — no bytes are copied or moved. The
 * resulting segment is only valid for as long as the underlying
 * ArrowBuf/vector is not closed or reallocated; callers must not retain a
 * returned segment past the lifetime of the vector it came from.
 * <p>
 * This uses a restricted method ({@code MemorySegment.ofAddress}
 * dereferencing an arbitrary native address) and requires the JVM flag
 * {@code --enable-native-access=ALL-UNNAMED} (or a module-qualified
 * equivalent) at runtime on JDK 22+, or you will hit a runtime warning
 * or {@code IllegalCallerException} depending on JDK version.
 */
public final class ArrowSegments {

    private ArrowSegments() {
    }

    /**
     * Zero-copy view of a {@link Float8Vector}'s data buffer as a
     * MemorySegment covering exactly {@code elementCount} doubles
     * ({@code elementCount * 8} bytes), starting at the vector's first
     * element.
     *
     * @param vector       the Arrow vector; must have at least {@code elementCount} allocated slots
     * @param elementCount number of doubles to expose (typically the batch's row count)
     */
    public static MemorySegment ofData(Float8Vector vector, long elementCount) {
        ArrowBuf buf = vector.getDataBuffer();
        long byteSize = elementCount * (long) Double.BYTES;
        if (buf.capacity() < byteSize) {
            throw new ArrowBindingException(
                    "Requested " + byteSize + " bytes (" + elementCount + " doubles) from vector '"
                            + safeName(vector) + "' but its data buffer only has " + buf.capacity()
                            + " bytes allocated. Did you call allocateNew(rowCount) / setValueCount(rowCount)?");
        }
        return MemorySegment.ofAddress(buf.memoryAddress()).reinterpret(byteSize);
    }

    /**
     * Zero-copy view of a vector's validity (null) bitmap as a
     * MemorySegment, covering exactly the bytes needed for {@code
     * elementCount} rows ({@code ceil(elementCount / 8)} bytes).
     */
    public static MemorySegment ofValidity(Float8Vector vector, long elementCount) {
        ArrowBuf buf = vector.getValidityBuffer();
        long byteSize = validityByteWidth(elementCount);
        if (buf.capacity() < byteSize) {
            throw new ArrowBindingException(
                    "Validity buffer for vector '" + safeName(vector) + "' has only " + buf.capacity()
                            + " bytes allocated but " + byteSize + " bytes are needed for " + elementCount
                            + " rows.");
        }
        return MemorySegment.ofAddress(buf.memoryAddress()).reinterpret(byteSize);
    }

    /** Bytes needed to hold a validity bitmap for {@code elementCount} rows (1 bit/row, byte-aligned). */
    public static long validityByteWidth(long elementCount) {
        return (elementCount + 7) / 8;
    }

    /**
     * Bitwise-ANDs {@code left} and {@code right} validity bitmaps into
     * {@code dest}, byte-wise — used by {@link NullPolicy#PROPAGATE_NULL}
     * to combine several input columns' null bitmaps into one output
     * bitmap without a per-row branch. All three segments must have the
     * same {@code byteSize()}; {@code dest} may alias {@code left} or
     * {@code right}.
     * <p>
     * This is a plain scalar byte loop, not SIMD-vectorized — the
     * validity bitmap is {@code rowCount / 8} bytes, i.e. tiny relative
     * to the {@code rowCount * 8}-byte data buffers the actual arithmetic
     * runs over, so it was not worth the added complexity for v1. A
     * {@code ByteVector}-based version is a reasonable follow-up if
     * profiling shows otherwise.
     */
    public static void andValidityInto(MemorySegment dest, MemorySegment left, MemorySegment right) {
        long n = dest.byteSize();
        if (left.byteSize() != n || right.byteSize() != n) {
            throw new IllegalArgumentException("andValidityInto: segment sizes must match ("
                    + left.byteSize() + ", " + right.byteSize() + ", " + n + ")");
        }
        for (long i = 0; i < n; i++) {
            byte l = left.get(ValueLayout.JAVA_BYTE, i);
            byte r = right.get(ValueLayout.JAVA_BYTE, i);
            dest.set(ValueLayout.JAVA_BYTE, i, (byte) (l & r));
        }
    }

    /** Copies {@code src} into {@code dest} unchanged — seeds an AND chain with the first column. */
    public static void copyValidityInto(MemorySegment dest, MemorySegment src) {
        MemorySegment.copy(src, 0, dest, 0, src.byteSize());
    }

    /**
     * Marks bits {@code [0, elementCount)} of a validity bitmap segment
     * as valid (non-null), byte-wise. Needed because a freshly {@code
     * allocateNew}'d Arrow vector's validity buffer starts out zeroed —
     * i.e. every row is null by default — and {@link ArrowBulkEvaluator}
     * writes its zero-copy results straight into the output vector's raw
     * data buffer via {@link #ofData}, bypassing the normal {@code
     * FieldVector.set(...)} calls that would otherwise flip each row's
     * validity bit. Any code path that computes real, non-null values
     * for every row this way (as opposed to {@link
     * NullPolicy#PROPAGATE_NULL}, which derives the bitmap from the
     * inputs instead) must call this afterward or every row will read
     * back as null despite holding a correct value.
     */
    public static void markAllValid(MemorySegment validity, long elementCount) {
        long fullBytes = elementCount / 8;
        for (long i = 0; i < fullBytes; i++) {
            validity.set(ValueLayout.JAVA_BYTE, i, (byte) 0xFF);
        }
        int remainderBits = (int) (elementCount - fullBytes * 8);
        if (remainderBits > 0) {
            byte mask = (byte) ((1 << remainderBits) - 1);
            validity.set(ValueLayout.JAVA_BYTE, fullBytes, mask);
        }
    }

    /**
     * Reads whether row {@code rowIndex} is non-null from a validity
     * bitmap segment (standard Arrow layout: bit {@code i} of byte
     * {@code i/8}, LSB-first — a set bit means non-null).
     */
    public static boolean isValid(MemorySegment validity, long rowIndex) {
        int byteIdx = (int) (rowIndex >> 3);
        int bitIdx = (int) (rowIndex & 7);
        byte b = validity.get(ValueLayout.JAVA_BYTE, byteIdx);
        return ((b >> bitIdx) & 1) != 0;
    }

    private static String safeName(Float8Vector vector) {
        try {
            return vector.getField().getName();
        } catch (RuntimeException e) {
            return "<unknown>";
        }
    }
}