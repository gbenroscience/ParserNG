package com.github.gbenroscience.arrow.tools.box;

import org.apache.arrow.memory.ArrowBuf;

import java.lang.foreign.MemorySegment;

/**
 * Bridges Apache Arrow's off-heap {@link ArrowBuf} buffers to the JDK
 * Foreign Function &amp; Memory API's {@link MemorySegment}, with zero copy
 * in either direction.
 *
 * <p><b>How this works:</b> {@link ArrowBuf} exposes the raw native address
 * of its backing memory via {@link ArrowBuf#memoryAddress()}. We wrap that
 * address as a zero-length {@link MemorySegment} via
 * {@link MemorySegment#ofAddress(long)}, then extend its declared bounds to
 * the byte length we intend to read via {@link MemorySegment#reinterpret(long)}.
 * This does not copy memory and does not take ownership of the buffer's
 * lifecycle — Arrow's own {@code ReferenceManager} remains solely
 * responsible for allocation and deallocation. The segment returned here is
 * only valid for as long as the source {@link ArrowBuf} has not been closed,
 * released, or reallocated by Arrow. Do not retain a segment returned by
 * this class past the lifetime of the {@link ArrowBuf} it was derived from.
 *
 * <p><b>Runtime requirement:</b> {@link MemorySegment#reinterpret(long)} is a
 * restricted FFM method. The JVM must be started with
 * {@code --enable-native-access=ALL-UNNAMED} (or the module-qualified
 * equivalent for this module) or calls here will throw at runtime.
 *
 * <p><b>Verification note:</b> {@code ArrowBuf.memoryAddress()} and
 * {@code ArrowBuf.capacity()} have been stable across many Arrow Java
 * releases, but this bridge relies on Arrow-internal memory layout
 * guarantees that this module cannot independently verify at compile time.
 * Cover this class with an integration test against the exact Arrow Java
 * version pinned in your {@code pom.xml} before trusting it in production —
 * ideally a round-trip test that writes known values into a real
 * {@code Float8Vector}, wraps it with {@link #wrapDoubles}, reads them back
 * through the returned segment, and asserts equality.
 */
public final class ArrowMemoryBridge {

    private ArrowMemoryBridge() {
    }

    /**
     * Wraps the first {@code elementCount} {@code double} elements of
     * {@code buf} as a zero-copy {@link MemorySegment}.
     *
     * @param buf          the Arrow buffer to wrap; must not be null
     * @param elementCount the number of {@code double} elements to expose,
     *                     starting at byte offset 0 of {@code buf}
     * @return a MemorySegment aliasing {@code buf}'s memory — no copy is made
     * @throws NullPointerException     if {@code buf} is null
     * @throws IllegalArgumentException if {@code elementCount} is negative or
     *                                  {@code buf} does not have enough capacity
     */
    public static MemorySegment wrapDoubles(ArrowBuf buf, long elementCount) {
        if (buf == null) {
            throw new NullPointerException("buf must not be null");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must not be negative, got " + elementCount);
        }
        long byteSize = elementCount * Double.BYTES;
        if (buf.capacity() < byteSize) {
            throw new IllegalArgumentException(
                "ArrowBuf capacity (" + buf.capacity() + " bytes) is smaller than the requested "
                    + elementCount + " doubles (" + byteSize + " bytes)");
        }
        return MemorySegment.ofAddress(buf.memoryAddress()).reinterpret(byteSize);
    }

    /**
     * Wraps the first {@code elementCount} {@code float} elements of
     * {@code buf} as a zero-copy {@link MemorySegment}.
     *
     * @param buf          the Arrow buffer to wrap; must not be null
     * @param elementCount the number of {@code float} elements to expose,
     *                     starting at byte offset 0 of {@code buf}
     * @return a MemorySegment aliasing {@code buf}'s memory — no copy is made
     * @throws NullPointerException     if {@code buf} is null
     * @throws IllegalArgumentException if {@code elementCount} is negative or
     *                                  {@code buf} does not have enough capacity
     */
    public static MemorySegment wrapFloats(ArrowBuf buf, long elementCount) {
        if (buf == null) {
            throw new NullPointerException("buf must not be null");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must not be negative, got " + elementCount);
        }
        long byteSize = elementCount * Float.BYTES;
        if (buf.capacity() < byteSize) {
            throw new IllegalArgumentException(
                "ArrowBuf capacity (" + buf.capacity() + " bytes) is smaller than the requested "
                    + elementCount + " floats (" + byteSize + " bytes)");
        }
        return MemorySegment.ofAddress(buf.memoryAddress()).reinterpret(byteSize);
    }

    /**
     * Wraps the full declared capacity of {@code buf} as a {@link MemorySegment},
     * for callers that already know they want the whole buffer rather than a
     * specific element count (e.g. validity bitmaps, which are sized in bytes
     * rather than doubles).
     */
    public static MemorySegment wrapFullCapacity(ArrowBuf buf) {
        if (buf == null) {
            throw new NullPointerException("buf must not be null");
        }
        return MemorySegment.ofAddress(buf.memoryAddress()).reinterpret(buf.capacity());
    }
}