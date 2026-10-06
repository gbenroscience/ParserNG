package com.github.gbenroscience.parser.ng.parquet.v1.internal.codec;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.Optional;

/**
 * Raw (unframed) Snappy block decompressor.
 *
 * <p>Implements the Snappy block format used by Parquet's {@code SNAPPY} codec: a varint preamble holding the
 * uncompressed length, followed by a sequence of literal and copy elements. The Snappy framing format
 * (stream identifier / CRC chunks) and the Hadoop block-framed variant are <b>not</b> handled.
 *
 * <h2>Two entry points, one algorithm</h2>
 * <ul>
 *   <li>{@link #decompress(byte[], int, int, byte[], int, int)} - heap arrays. This is the fastest path
 *       (word access through byte-array {@link VarHandle}s) and the one parquet-java's heap
 *       {@code ByteBuffer}s should use.</li>
 *   <li>{@link #decompress(MemorySegment, long, long, MemorySegment, long, long)} - any segment (heap, native,
 *       mapped). Segments that are backed by a writable {@code byte[]} on both sides are detected and routed to
 *       the array path automatically; everything else runs the equivalent {@code MemorySegment} loop.</li>
 * </ul>
 *
 * <h2>How it is fast</h2>
 * Decoding runs in two phases over the same element grammar:
 * <ol>
 *   <li><b>Fast loop</b>, while at least {@link #SLOP} bytes remain in <em>both</em> streams. Every element
 *       provably fits, so literals and back-references move with unconditional 8-byte word copies that
 *       deliberately over-copy into space a later element overwrites (the same trick as libsnappy's
 *       {@code UnalignedCopy64}/{@code IncrementalCopy}). The three copy encodings are decoded branch-free
 *       through a 256-entry table, because the element mix is effectively random on real data and each
 *       mispredicted branch costs ~15-20 cycles. Short-period overlaps (offset &lt; 8) are expanded from a
 *       register-built 8-byte pattern.</li>
 *   <li><b>Tail loop</b> for the last few dozen bytes: byte-exact with full per-element validation, so no
 *       scratch write ever lands beyond the output region.</li>
 * </ol>
 *
 * <h2>Guarantees</h2>
 * <ul>
 *   <li>All input is untrusted. Corrupt input yields {@link CorruptInputException}; there is no out-of-bounds
 *       access, silent truncation or unbounded allocation. All writes (including scratch over-copies) stay
 *       inside {@code [dstOffset, dstOffset + uncompressedLength)}; on success every byte there is final. On
 *       failure the contents of that region are unspecified.</li>
 *   <li>Overlapping back-references (offset &lt; length) have the format's run-length semantics.</li>
 *   <li>Stateless and thread-safe. {@code src} and {@code dst} must not overlap in memory.</li>
 * </ul>
 */
public final class SnappyDecompressor {

    /** Thrown when the compressed input is malformed. */
    public static final class CorruptInputException extends IOException {
        private static final long serialVersionUID = 1L;

        public CorruptInputException(String message) {
            super(message);
        }
    }

    // ---- MemorySegment layouts -------------------------------------------------------------------
    private static final ValueLayout.OfByte BYTE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort LE_SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    // ---- byte[] views (bounds-checked by the JIT; unaligned little-endian) -----------------------
    private static final VarHandle A_LONG =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final VarHandle A_INT =
            MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);
    private static final VarHandle A_SHORT =
            MethodHandles.byteArrayViewVarHandle(short[].class, ByteOrder.LITTLE_ENDIAN);

    /** Max varint length for a 32-bit uncompressed size. */
    private static final int MAX_VARINT_BYTES = 5;

    /**
     * Bytes of headroom required, in both streams, before an element may take the fast path. The largest
     * element touches 1 tag byte + up to 64 literal bytes (rounded up to whole words) on input, and writes up
     * to 64 bytes (rounded up to whole words) on output.
     */
    private static final int SLOP = 72;

    /**
     * Per-tag decode table for COPY tags ({@code tag & 3 != 0}), like libsnappy's {@code char_table}:
     * bits 0-7 = copy length, bits 8-10 = high offset bits (copy-1 only), bits 11-13 = number of trailing
     * offset bytes (1, 2 or 4).
     */
    private static final int[] COPY_ENTRY = new int[256];

    static {
        for (int tag = 0; tag < 256; tag++) {
            switch (tag & 3) {
                case 1 -> COPY_ENTRY[tag] = (4 + ((tag >>> 2) & 7)) | (((tag >>> 5) & 7) << 8) | (1 << 11);
                case 2 -> COPY_ENTRY[tag] = (1 + (tag >>> 2)) | (2 << 11);
                case 3 -> COPY_ENTRY[tag] = (1 + (tag >>> 2)) | (4 << 11);
                default -> COPY_ENTRY[tag] = 0; // literal: not used
            }
        }
    }

    private SnappyDecompressor() {}

    // ==========================================================================================
    // Public API
    // ==========================================================================================

    /** Reads the uncompressed length from the Snappy preamble without decompressing. */
    public static long uncompressedLength(MemorySegment src, long srcOffset, long srcLimit)
            throws CorruptInputException {
        Objects.requireNonNull(src, "src");
        checkRange(srcOffset, srcLimit, src.byteSize(), "src");
        return readPreamble(src, srcOffset, srcLimit) >>> 3;
    }

    /** Reads the uncompressed length from the Snappy preamble without decompressing. */
    public static long uncompressedLength(byte[] src, int srcOffset, int srcLimit)
            throws CorruptInputException {
        Objects.requireNonNull(src, "src");
        checkRange(srcOffset, srcLimit, src.length, "src");
        return readPreamble(src, srcOffset, srcLimit) >>> 3;
    }

    /**
     * Decompresses a Snappy block held in heap arrays (the fastest entry point).
     *
     * @return number of bytes written to {@code dst}; always equals the preamble's length
     * @throws CorruptInputException    if the input is malformed or truncated
     * @throws IllegalArgumentException if the destination region is smaller than the declared
     *                                  uncompressed length, or the ranges are invalid
     */
    public static int decompress(byte[] src, int srcOffset, int srcLimit,
                                 byte[] dst, int dstOffset, int dstLimit)
            throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(dst, "dst");
        checkRange(srcOffset, srcLimit, src.length, "src");
        checkRange(dstOffset, dstLimit, dst.length, "dst");

        final long preamble = readPreamble(src, srcOffset, srcLimit);
        final long uncompressedLength = preamble >>> 3;
        int sp = srcOffset + (int) (preamble & 7);

        // Validate capacity up front so a hostile length can never cause a partial write.
        if (uncompressedLength > (long) dstLimit - dstOffset) {
            throw new IllegalArgumentException("Destination too small: need " + uncompressedLength
                    + " bytes, have " + ((long) dstLimit - dstOffset));
        }
        final int dstEnd = dstOffset + (int) uncompressedLength;
        int dp = dstOffset;

        // ---- Phase 1: fast loop -----------------------------------------------------------------
        final int srcFast = srcLimit - SLOP;
        final int dstFast = dstEnd - SLOP;
        while (sp <= srcFast && dp <= dstFast) {
            final int tag = src[sp] & 0xFF;

            if ((tag & 0x03) == 0x00) {
                // ---- LITERAL ----
                final int code = tag >>> 2;
                if (code < 60) {
                    // 1..60 bytes, length in the tag. Copy 16 bytes unconditionally, then more words if needed.
                    final int len = code + 1;
                    A_LONG.set(dst, dp, (long) A_LONG.get(src, sp + 1));
                    A_LONG.set(dst, dp + 8, (long) A_LONG.get(src, sp + 9));
                    if (len > 16) {
                        for (int i = 16; i < len; i += 8) {
                            A_LONG.set(dst, dp + i, (long) A_LONG.get(src, sp + 1 + i));
                        }
                    }
                    sp += len + 1;
                    dp += len;
                } else {
                    // 61..2^32 bytes, length in 1..4 trailing bytes (SLOP guarantees they are present).
                    final int extra = code - 59;
                    final long raw = ((int) A_INT.get(src, sp + 1)) & 0xFFFFFFFFL;
                    final long len = (raw & (0xFFFFFFFFL >>> (32 - 8 * extra))) + 1;
                    sp += 1 + extra;
                    if (len > srcLimit - sp) {
                        throw corrupt("Literal of " + len + " bytes overruns input at src offset " + sp);
                    }
                    if (len > dstEnd - dp) {
                        throw corrupt("Literal of " + len + " bytes exceeds declared uncompressed length");
                    }
                    System.arraycopy(src, sp, dst, dp, (int) len);
                    sp += (int) len;
                    dp += (int) len;
                }
                continue;
            }

            // ---- COPY (branch-free decode of all three encodings) ----
            final int entry = COPY_ENTRY[tag];
            final int len = entry & 0xFF;
            final int extra = entry >>> 11; // 1, 2 or 4 trailing offset bytes
            final long offsetL = (entry & 0x700)
                    + (((int) A_INT.get(src, sp + 1)) & ((1L << (extra << 3)) - 1));
            sp += 1 + extra;

            // Offset is relative to this block's output start, never to the array start. One unsigned
            // compare rejects both offset == 0 (offset-1 wraps) and offset > bytes written so far.
            // (len <= 64 <= SLOP, so the output-capacity check is implied by the loop condition.)
            if (Long.compareUnsigned(offsetL - 1, dp - dstOffset) >= 0) {
                throw badOffset(offsetL, dp - dstOffset);
            }
            final int offset = (int) offsetL; // <= bytes written, so fits an int
            if (len <= 16 && offset >= 8) {
                // Dominant case (libsnappy measures 70-80% of copies): two plain word moves.
                final int m = dp - offset;
                A_LONG.set(dst, dp, (long) A_LONG.get(dst, m));
                A_LONG.set(dst, dp + 8, (long) A_LONG.get(dst, m + 8));
            } else {
                copyMatchFast(dst, dp, offset, len);
            }
            dp += len;
        }

        // ---- Phase 2: exact tail loop -------------------------------------------------------------
        while (sp < srcLimit) {
            final int tag = src[sp++] & 0xFF;

            switch (tag & 0x03) {
                case 0x00 -> {
                    long len = tag >>> 2;
                    if (len < 60) {
                        len += 1;
                    } else {
                        final int extra = (int) (len - 59);
                        if (srcLimit - sp < extra) {
                            throw corrupt("Truncated literal length at src offset " + sp);
                        }
                        long v = 0;
                        for (int i = 0; i < extra; i++) {
                            v |= (long) (src[sp++] & 0xFF) << (8 * i);
                        }
                        len = v + 1;
                    }
                    if (len > srcLimit - sp) {
                        throw corrupt("Literal of " + len + " bytes overruns input at src offset " + sp);
                    }
                    if (len > dstEnd - dp) {
                        throw corrupt("Literal of " + len + " bytes exceeds declared uncompressed length");
                    }
                    System.arraycopy(src, sp, dst, dp, (int) len);
                    sp += (int) len;
                    dp += (int) len;
                }
                case 0x01 -> {
                    if (srcLimit - sp < 1) {
                        throw corrupt("Truncated copy-1 element at src offset " + sp);
                    }
                    final long len = 4 + ((tag >>> 2) & 0x07);
                    final long offset = ((long) (tag & 0xE0) << 3) | (src[sp++] & 0xFF);
                    dp = copyBackReferenceExact(dst, dstOffset, dp, dstEnd, offset, len);
                }
                case 0x02 -> {
                    if (srcLimit - sp < 2) {
                        throw corrupt("Truncated copy-2 element at src offset " + sp);
                    }
                    final long len = 1 + (tag >>> 2);
                    final long offset = (src[sp] & 0xFF) | ((src[sp + 1] & 0xFF) << 8);
                    sp += 2;
                    dp = copyBackReferenceExact(dst, dstOffset, dp, dstEnd, offset, len);
                }
                default -> {
                    if (srcLimit - sp < 4) {
                        throw corrupt("Truncated copy-4 element at src offset " + sp);
                    }
                    final long len = 1 + (tag >>> 2);
                    final long offset = ((int) A_INT.get(src, sp)) & 0xFFFFFFFFL; // unsigned
                    sp += 4;
                    dp = copyBackReferenceExact(dst, dstOffset, dp, dstEnd, offset, len);
                }
            }
        }

        if (dp != dstEnd) {
            throw corrupt("Decompressed " + (dp - dstOffset) + " bytes but preamble declared "
                    + uncompressedLength);
        }
        return (int) uncompressedLength;
    }

    /**
     * Decompresses a Snappy block held in memory segments.
     *
     * <p>If both segments are backed by a writable {@code byte[]}, this transparently uses the faster array
     * path; native, mapped and read-only segments use the {@code MemorySegment} loop below.
     *
     * @return number of bytes written to {@code dst}; always equals the preamble's length
     * @throws CorruptInputException    if the input is malformed or truncated
     * @throws IllegalArgumentException if the destination region is smaller than the declared
     *                                  uncompressed length, or the ranges are invalid
     */
    public static long decompress(MemorySegment src, long srcOffset, long srcLimit,
                                  MemorySegment dst, long dstOffset, long dstLimit)
            throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(dst, "dst");
        checkRange(srcOffset, srcLimit, src.byteSize(), "src");
        checkRange(dstOffset, dstLimit, dst.byteSize(), "dst");
        if (dst.isReadOnly()) {
            throw new IllegalArgumentException("dst is read-only");
        }

        // Route heap-backed segments to the array path (no copying: same arrays, translated offsets).
        final byte[] srcArr = backingArray(src);
        if (srcArr != null) {
            final byte[] dstArr = backingArray(dst);
            if (dstArr != null) {
                final long sBase = src.address();
                final long dBase = dst.address();
                if (fitsInt(sBase + srcLimit) && fitsInt(dBase + dstLimit)) {
                    return decompress(srcArr, (int) (sBase + srcOffset), (int) (sBase + srcLimit),
                            dstArr, (int) (dBase + dstOffset), (int) (dBase + dstLimit));
                }
            }
        }

        final long preamble = readPreamble(src, srcOffset, srcLimit);
        final long uncompressedLength = preamble >>> 3;
        long srcPos = srcOffset + (preamble & 7);

        if (uncompressedLength > dstLimit - dstOffset) {
            throw new IllegalArgumentException("Destination too small: need " + uncompressedLength
                    + " bytes, have " + (dstLimit - dstOffset));
        }

        final long dstEnd = dstOffset + uncompressedLength;
        long dstPos = dstOffset;

        // ---- Phase 1: fast loop (see the array version for the commentary) -------------------------
        final long srcFast = srcLimit - SLOP;
        final long dstFast = dstEnd - SLOP;
        while (srcPos <= srcFast && dstPos <= dstFast) {
            final int tag = src.get(BYTE, srcPos) & 0xFF;

            if ((tag & 0x03) == 0x00) {
                final int code = tag >>> 2;
                if (code < 60) {
                    final int len = code + 1;
                    dst.set(LE_LONG, dstPos, src.get(LE_LONG, srcPos + 1));
                    dst.set(LE_LONG, dstPos + 8, src.get(LE_LONG, srcPos + 9));
                    if (len > 16) {
                        for (int i = 16; i < len; i += 8) {
                            dst.set(LE_LONG, dstPos + i, src.get(LE_LONG, srcPos + 1 + i));
                        }
                    }
                    srcPos += len + 1;
                    dstPos += len;
                } else {
                    final int extra = code - 59;
                    final long raw = src.get(LE_INT, srcPos + 1) & 0xFFFFFFFFL;
                    final long len = (raw & (0xFFFFFFFFL >>> (32 - 8 * extra))) + 1;
                    srcPos += 1 + extra;
                    if (len > srcLimit - srcPos) {
                        throw corrupt("Literal of " + len + " bytes overruns input at src offset " + srcPos);
                    }
                    if (len > dstEnd - dstPos) {
                        throw corrupt("Literal of " + len + " bytes exceeds declared uncompressed length");
                    }
                    MemorySegment.copy(src, srcPos, dst, dstPos, len);
                    srcPos += len;
                    dstPos += len;
                }
                continue;
            }

            final int entry = COPY_ENTRY[tag];
            final int len = entry & 0xFF;
            final int extra = entry >>> 11;
            final long offset = (entry & 0x700)
                    + (src.get(LE_INT, srcPos + 1) & ((1L << (extra << 3)) - 1));
            srcPos += 1 + extra;

            if (Long.compareUnsigned(offset - 1, dstPos - dstOffset) >= 0) {
                throw badOffset(offset, dstPos - dstOffset);
            }
            if (len <= 16 && offset >= 8) {
                final long m = dstPos - offset;
                dst.set(LE_LONG, dstPos, dst.get(LE_LONG, m));
                dst.set(LE_LONG, dstPos + 8, dst.get(LE_LONG, m + 8));
            } else {
                copyMatchFast(dst, dstPos, offset, len);
            }
            dstPos += len;
        }

        // ---- Phase 2: exact tail loop -------------------------------------------------------------
        while (srcPos < srcLimit) {
            final int tag = src.get(BYTE, srcPos++) & 0xFF;

            switch (tag & 0x03) {
                case 0x00 -> {
                    long len = tag >>> 2;
                    if (len < 60) {
                        len += 1;
                    } else {
                        final int extra = (int) (len - 59);
                        if (srcLimit - srcPos < extra) {
                            throw corrupt("Truncated literal length at src offset " + srcPos);
                        }
                        long v = 0;
                        for (int i = 0; i < extra; i++) {
                            v |= (long) (src.get(BYTE, srcPos++) & 0xFF) << (8 * i);
                        }
                        len = v + 1;
                    }
                    if (len > srcLimit - srcPos) {
                        throw corrupt("Literal of " + len + " bytes overruns input at src offset " + srcPos);
                    }
                    if (len > dstEnd - dstPos) {
                        throw corrupt("Literal of " + len + " bytes exceeds declared uncompressed length");
                    }
                    MemorySegment.copy(src, srcPos, dst, dstPos, len);
                    srcPos += len;
                    dstPos += len;
                }
                case 0x01 -> {
                    if (srcLimit - srcPos < 1) {
                        throw corrupt("Truncated copy-1 element at src offset " + srcPos);
                    }
                    final long len = 4 + ((tag >>> 2) & 0x07);
                    final long offset = ((long) (tag & 0xE0) << 3) | (src.get(BYTE, srcPos++) & 0xFF);
                    dstPos = copyBackReferenceExact(dst, dstOffset, dstPos, dstEnd, offset, len);
                }
                case 0x02 -> {
                    if (srcLimit - srcPos < 2) {
                        throw corrupt("Truncated copy-2 element at src offset " + srcPos);
                    }
                    final long len = 1 + (tag >>> 2);
                    final long offset = src.get(LE_SHORT, srcPos) & 0xFFFFL;
                    srcPos += 2;
                    dstPos = copyBackReferenceExact(dst, dstOffset, dstPos, dstEnd, offset, len);
                }
                default -> {
                    if (srcLimit - srcPos < 4) {
                        throw corrupt("Truncated copy-4 element at src offset " + srcPos);
                    }
                    final long len = 1 + (tag >>> 2);
                    final long offset = src.get(LE_INT, srcPos) & 0xFFFFFFFFL; // unsigned
                    srcPos += 4;
                    dstPos = copyBackReferenceExact(dst, dstOffset, dstPos, dstEnd, offset, len);
                }
            }
        }

        if (dstPos != dstEnd) {
            throw corrupt("Decompressed " + (dstPos - dstOffset) + " bytes but preamble declared "
                    + uncompressedLength);
        }
        return uncompressedLength;
    }

    /** Convenience overload using the whole of both segments. */
    public static long decompress(MemorySegment src, MemorySegment dst) throws IOException {
        return decompress(src, 0, src.byteSize(), dst, 0, dst.byteSize());
    }

    // ==========================================================================================
    // Internals
    // ==========================================================================================

    /** @return the {@code byte[]} behind a writable heap segment, or {@code null} for anything else. */
    private static byte[] backingArray(MemorySegment s) {
        final Optional<Object> base = s.heapBase(); // empty for native and for read-only segments
        if (base.isPresent() && base.get() instanceof byte[] a) {
            final long address = s.address(); // for heap segments: byte offset into the array
            if (address >= 0 && address + s.byteSize() <= a.length) {
                return a;
            }
        }
        return null;
    }

    private static boolean fitsInt(long v) {
        return v >= 0 && v <= Integer.MAX_VALUE;
    }

    /**
     * Fast-path back-reference copy. Preconditions (established by the caller): {@code 1 <= len <= 64},
     * {@code 1 <= offset <= dp - dstOffset}, and at least {@link #SLOP} writable bytes at {@code dp}.
     * Writes the {@code len} correct bytes and may scribble up to 7 further bytes inside the slop.
     */
    private static void copyMatchFast(byte[] dst, int dp, int offset, int len) {
        final int m = dp - offset;
        if (offset >= 8) {
            // Each 8-byte chunk reads only bytes already final (m + 8 + i <= dp + i), so overlap with
            // 8 <= offset < len is handled by plain forward chunking.
            A_LONG.set(dst, dp, (long) A_LONG.get(dst, m));
            if (len > 8) {
                A_LONG.set(dst, dp + 8, (long) A_LONG.get(dst, m + 8));
                for (int i = 16; i < len; i += 8) {
                    A_LONG.set(dst, dp + i, (long) A_LONG.get(dst, m + i));
                }
            }
            return;
        }
        // offset < 8: the output repeats a pattern of period `offset`. Periods 1, 2, 4 divide 8, so one
        // register-built pattern word is stamped repeatedly; periods 3, 5, 6, 7 write 8 bytes forward, then
        // stamp words from the smallest multiple of the period that is >= 8 back.
        final long pattern;
        switch (offset) {
            case 1 -> pattern = (dst[m] & 0xFFL) * 0x0101010101010101L;
            case 2 -> pattern = (((short) A_SHORT.get(dst, m)) & 0xFFFFL) * 0x0001000100010001L;
            case 4 -> pattern = (((int) A_INT.get(dst, m)) & 0xFFFFFFFFL) * 0x0000000100000001L;
            default -> {
                for (int i = 0; i < 8; i++) {
                    dst[dp + i] = dst[m + i];
                }
                if (len > 8) {
                    final int eo = offset * ((8 + offset - 1) / offset);
                    for (int i = 8; i < len; i += 8) {
                        A_LONG.set(dst, dp + i, (long) A_LONG.get(dst, dp + i - eo));
                    }
                }
                return;
            }
        }
        for (int i = 0; i < len; i += 8) {
            A_LONG.set(dst, dp + i, pattern);
        }
    }

    /** MemorySegment twin of {@link #copyMatchFast(byte[], int, int, int)}. */
    private static void copyMatchFast(MemorySegment dst, long dp, long offset, int len) {
        final long m = dp - offset;
        if (offset >= 8) {
            dst.set(LE_LONG, dp, dst.get(LE_LONG, m));
            if (len > 8) {
                dst.set(LE_LONG, dp + 8, dst.get(LE_LONG, m + 8));
                for (int i = 16; i < len; i += 8) {
                    dst.set(LE_LONG, dp + i, dst.get(LE_LONG, m + i));
                }
            }
            return;
        }
        final long pattern;
        switch ((int) offset) {
            case 1 -> pattern = (dst.get(BYTE, m) & 0xFFL) * 0x0101010101010101L;
            case 2 -> pattern = (dst.get(LE_SHORT, m) & 0xFFFFL) * 0x0001000100010001L;
            case 4 -> pattern = (dst.get(LE_INT, m) & 0xFFFFFFFFL) * 0x0000000100000001L;
            default -> {
                for (int i = 0; i < 8; i++) {
                    dst.set(BYTE, dp + i, dst.get(BYTE, m + i));
                }
                if (len > 8) {
                    final long eo = offset * ((8 + offset - 1) / offset);
                    for (int i = 8; i < len; i += 8) {
                        dst.set(LE_LONG, dp + i, dst.get(LE_LONG, dp + i - eo));
                    }
                }
                return;
            }
        }
        for (int i = 0; i < len; i += 8) {
            dst.set(LE_LONG, dp + i, pattern);
        }
    }

    /** Exact (byte-bounded) back-reference copy used by the tail loops. */
    private static int copyBackReferenceExact(byte[] dst, int dstOffset, int dp, int dstEnd,
                                              long offset, long len) throws CorruptInputException {
        validateBackReference(dp - dstOffset, dstEnd - dp, offset, len);
        final int m = dp - (int) offset;
        final int n = (int) len;
        // Forward byte-wise copy gives the exact overlap semantics for free; this only runs for the last
        // ~SLOP bytes of a block, so simplicity beats speed.
        for (int i = 0; i < n; i++) {
            dst[dp + i] = dst[m + i];
        }
        return dp + n;
    }

    private static long copyBackReferenceExact(MemorySegment dst, long dstOffset, long dstPos, long dstEnd,
                                               long offset, long len) throws CorruptInputException {
        validateBackReference(dstPos - dstOffset, dstEnd - dstPos, offset, len);
        final long matchPos = dstPos - offset;
        for (long i = 0; i < len; i++) {
            dst.set(BYTE, dstPos + i, dst.get(BYTE, matchPos + i));
        }
        return dstPos + len;
    }

    private static void validateBackReference(long written, long remaining, long offset, long len)
            throws CorruptInputException {
        if (offset == 0) {
            throw corrupt("Copy with zero offset");
        }
        // Offset is relative to this block's output start, never to the buffer start.
        if (offset > written) {
            throw corrupt("Copy offset " + offset + " reaches before start of output (" + written
                    + " bytes written)");
        }
        if (len > remaining) {
            throw corrupt("Copy of " + len + " bytes exceeds declared uncompressed length");
        }
    }

    /** @return {@code (uncompressedLength << 3) | bytesConsumedByPreamble}; allocation-free. */
    private static long readPreamble(MemorySegment src, long srcOffset, long srcLimit)
            throws CorruptInputException {
        long result = 0;
        long pos = srcOffset;
        for (int i = 0; i < MAX_VARINT_BYTES; i++) {
            if (pos >= srcLimit) {
                throw corrupt("Truncated or missing length preamble");
            }
            final int b = src.get(BYTE, pos++) & 0xFF;
            // The 5th byte may only contribute 4 bits (total 32 bits) and cannot continue.
            if (i == MAX_VARINT_BYTES - 1 && (b & 0xF0) != 0) {
                throw corrupt("Uncompressed length exceeds 32 bits");
            }
            result |= (long) (b & 0x7F) << (7 * i);
            if ((b & 0x80) == 0) {
                return (result << 3) | (pos - srcOffset);
            }
        }
        throw corrupt("Length preamble too long");
    }

    private static long readPreamble(byte[] src, int srcOffset, int srcLimit) throws CorruptInputException {
        long result = 0;
        int pos = srcOffset;
        for (int i = 0; i < MAX_VARINT_BYTES; i++) {
            if (pos >= srcLimit) {
                throw corrupt("Truncated or missing length preamble");
            }
            final int b = src[pos++] & 0xFF;
            if (i == MAX_VARINT_BYTES - 1 && (b & 0xF0) != 0) {
                throw corrupt("Uncompressed length exceeds 32 bits");
            }
            result |= (long) (b & 0x7F) << (7 * i);
            if ((b & 0x80) == 0) {
                return (result << 3) | (pos - srcOffset);
            }
        }
        throw corrupt("Length preamble too long");
    }

    private static void checkRange(long offset, long limit, long size, String name) {
        if (offset < 0 || limit < offset || limit > size) {
            throw new IllegalArgumentException(
                    name + " range [" + offset + ", " + limit + ") invalid for segment of size " + size);
        }
    }

    private static CorruptInputException badOffset(long offset, long written) {
        return offset == 0
                ? corrupt("Copy with zero offset")
                : corrupt("Copy offset " + offset + " reaches before start of output (" + written
                        + " bytes written)");
    }

    private static CorruptInputException corrupt(String message) {
        return new CorruptInputException("Corrupt Snappy input: " + message);
    }
}
