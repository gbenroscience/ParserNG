package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * Mutable cursor over a {@code byte[]} page buffer. Every method advances {@link #pos} and returns
 * a primitive; nothing here allocates. Replaces pulling values one at a time through parquet-column's
 * {@code ValuesReader}/{@code ColumnReader} chain, which boxes and wraps per call.
 *
 * <h2>Bounds checking</h2>
 * Every read validates {@code pos} against {@link #limit} before touching {@link #buf} and throws
 * {@link Corrupt} on failure, rather than reading raw array elements unchecked. This is a correctness
 * requirement, not a defensive nicety: every byte this class reads ultimately comes from a page that
 * was produced outside this process (on disk, possibly by a different writer, possibly corrupted in
 * transit, possibly decoded by a library path elsewhere in this module that has its own bug). Once a
 * decode desyncs -- for any reason -- {@code pos} can end up arbitrarily far past {@link #limit}. Left
 * unchecked, that desync is invisible until it happens to walk off the end of the *physical* backing
 * array (not the logical page), which can be many megabytes and many unrelated reads later, and by
 * then the exception (a raw {@code ArrayIndexOutOfBoundsException} with a huge, meaningless index) is
 * reported from whatever read happened to first cross the physical array boundary -- almost always a
 * completely different column/page/thread than the one that actually went wrong. Bounds-checking at
 * {@link #limit} turns that into an exception thrown at the first byte past the *logical* page, with
 * {@code pos}/{@code limit} attached, at the actual site of the desync. Callers are expected to catch
 * {@link Corrupt} and rethrow with file/column/row-group context (see {@code FastColumnCursor} and
 * {@code RowGroupDecoder#begin}); it deliberately carries no such context itself, since this class has
 * no notion of which file, column, or row group it is decoding for.
 */
public final class ByteReader {

    public byte[] buf;
    public int pos;
    public int limit;

    /** Thrown the instant a read would go past {@link #limit}. Carries only cursor state -- no file/column
     *  context, which callers with that context are expected to attach when they catch and rethrow. */
    public static final class Corrupt extends RuntimeException {
        public final int pos, need, limit, capacity;
        Corrupt(int pos, int need, int limit, int capacity) {
            super("page buffer underrun: need " + need + " byte(s) at pos=" + pos
                    + " but logical limit=" + limit + " (physical capacity=" + capacity + ")");
            this.pos = pos; this.need = need; this.limit = limit; this.capacity = capacity;
        }
    }

    public void wrap(byte[] buf, int offset, int length) {
        this.buf = buf;
        this.pos = offset;
        this.limit = offset + length;
    }

    public boolean hasRemaining() { return pos < limit; }
    public int remaining() { return limit - pos; }

    private void require(int nBytes) {
        if (pos + nBytes > limit || pos < 0) throw new Corrupt(pos, nBytes, limit, buf.length);
    }

    public int readIntLE() {
        require(4);
        int v = (buf[pos] & 0xFF) | ((buf[pos + 1] & 0xFF) << 8) | ((buf[pos + 2] & 0xFF) << 16) | ((buf[pos + 3] & 0xFF) << 24);
        pos += 4;
        return v;
    }

    public long readLongLE() {
        long lo = readIntLE() & 0xFFFFFFFFL;
        long hi = readIntLE() & 0xFFFFFFFFL;
        return (hi << 32) | lo;
    }

    public float readFloatLE() { return Float.intBitsToFloat(readIntLE()); }
    public double readDoubleLE() { return Double.longBitsToDouble(readLongLE()); }

    public byte readByte() { require(1); return buf[pos++]; }

    /** ULEB128, as used for RLE run headers. */
    public int readUnsignedVarInt() {
        int v = 0, shift = 0, b;
        do {
            require(1);
            b = buf[pos++] & 0xFF;
            v |= (b & 0x7F) << shift;
            shift += 7;
            if (shift > 35) throw new Corrupt(pos, 1, limit, buf.length); // malformed: no terminating byte within a sane width
        } while ((b & 0x80) != 0);
        return v;
    }

    /**
     * ULEB128 into a {@code long}, for DELTA_BINARY_PACKED header/block fields (block size, miniblock
     * count and value count fit comfortably in an {@code int} and could use {@link #readUnsignedVarInt()},
     * but the zigzag-encoded first-value/min-delta fields are full 64-bit quantities for INT64 columns).
     */
    public long readUnsignedVarLong() {
        long v = 0;
        int shift = 0, b;
        do {
            require(1);
            b = buf[pos++] & 0xFF;
            v |= (long) (b & 0x7F) << shift;
            shift += 7;
            if (shift > 70) throw new Corrupt(pos, 1, limit, buf.length); // malformed: no terminating byte within a sane width
        } while ((b & 0x80) != 0);
        return v;
    }

    /** {@code (n >>> 1) ^ -(n & 1)} -- standard zigzag decode, shared by every DELTA_* decoder. */
    public static long zigZagDecode(long n) {
        return (n >>> 1) ^ -(n & 1);
    }

    /**
     * Generic LSB-first bit-unpacking for widths up to 64, into a {@code long[]}. Used only by the
     * DELTA_* decoders (see {@code DeltaBinaryPackedDecoder}), whose miniblocks can carry up to
     * 64-bit-wide unsigned deltas for INT64 columns -- wider than {@link #readBitPackedGroups} (int[],
     * capped at 32 bits) supports. Correctness-first: a straightforward byte-at-a-time unpack rather
     * than {@link #readBitPackedGroups}'s vectorized 64-bit-load fast paths, since this path has not
     * been benchmarked (see the class Javadoc's general bounds-checking rationale, which applies here
     * identically: every byte still comes from an untrusted page).
     *
     * <p>Always consumes exactly {@code groupCount8 * 8} values' worth of bytes ({@code groupCount8 *
     * bitWidth} bytes), even when {@code maxWrite} is smaller -- matching {@link #readBitPackedGroups}'s
     * contract, and required here because a DELTA_BINARY_PACKED miniblock is always fully present in the
     * stream (padded with zeros) regardless of how many of its values a final, partial block actually uses.
     */
    public void readBitPackedGroupsLong(int bitWidth, int groupCount8, long[] out, int offset, int maxWrite) {
        final long totalValues = (long) groupCount8 * 8L;
        final int toWrite = (int) Math.min(totalValues, Math.max(maxWrite, 0));
        if (bitWidth == 0) {
            java.util.Arrays.fill(out, offset, offset + toWrite, 0L);
            return;
        }
        if (bitWidth < 0 || bitWidth > 64) {
            throw new IllegalArgumentException("invalid bit width " + bitWidth);
        }
        final long needBytes = (long) groupCount8 * bitWidth;
        if (pos < 0 || needBytes > (long) limit - pos) {
            throw new Corrupt(pos, (int) Math.min(needBytes, Integer.MAX_VALUE), limit, buf.length);
        }
        final byte[] b = buf;
        int bytePos = pos, bitPos = 0, o = offset;
        for (int v = 0; v < toWrite; v++) {
            long value = 0;
            int bitsFilled = 0;
            int bp = bytePos, bitp = bitPos;
            while (bitsFilled < bitWidth) {
                int avail = 8 - bitp;
                int take = Math.min(avail, bitWidth - bitsFilled);
                int chunk = (b[bp] & 0xFF) >>> bitp;
                chunk &= (take == 8) ? 0xFF : ((1 << take) - 1);
                value |= ((long) chunk) << bitsFilled;
                bitsFilled += take;
                bitp += take;
                if (bitp == 8) { bitp = 0; bp++; }
            }
            out[o++] = value;
            final int totalBit = bitPos + bitWidth;
            bytePos += totalBit / 8;
            bitPos = totalBit % 8;
        }
        pos += (int) needBytes; // skip the whole run, including any values beyond maxWrite -- see contract note above
    }

    /** Little-endian 64-bit view over a {@code byte[]}: one load covers a whole 8-value group for bit widths 2..8. */
    private static final VarHandle LONG_LE = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /**
     * Reads {@code groupCount8 * 8} bit-packed values from the input, in order -- but writes only
     * the first {@code maxWrite} of them into {@code out[offset..offset+maxWrite)}; any beyond that
     * are skipped (never decoded, never written) and {@link #pos} is advanced past them so whatever
     * follows in the page is still read from the right place.
     *
     * <p>This bound is a correctness requirement, not an optimization: {@code groupCount8} comes
     * directly from an unsigned varint in the file (the bit-packed run header) and is never validated
     * against the page's own remaining value count before this method runs. A well-formed encoder never
     * writes a run larger than needed, but the format does not forbid it, and every Parquet file here is
     * treated as untrusted input. No run size, however large, can write outside {@code out} or force an
     * allocation sized by an attacker-controlled value.
     *
     * <h2>Bounds checking: once per run, not once per byte</h2>
     * A run of {@code groupCount8} groups occupies exactly {@code groupCount8 * bitWidth} input bytes
     * (8 values of {@code bitWidth} bits each, per group). That length is computed in {@code long}
     * arithmetic (so a hostile {@code groupCount8} cannot overflow {@code int}) and checked against
     * {@link #limit} <em>once, up front</em>; if the run does not fit, {@link Corrupt} is thrown before a
     * single byte is consumed. After that check every byte the loops below touch is known to lie inside
     * the logical page, so the inner loops carry no per-byte {@code require}. The observable contract is
     * unchanged from the per-byte version (a truncated run throws {@link Corrupt}); it is simply reported
     * at the start of the run rather than at the first missing byte.
     *
     * <h2>Fast paths</h2>
     * <ul>
     *   <li>{@code bitWidth == 0}: no input bytes, zero fill.</li>
     *   <li>{@code bitWidth == 1} (definition levels of an optional flat column, boolean-like
     *       dictionaries): one input byte yields 8 values by shift-and-mask, no bit buffer.</li>
     *   <li>{@code bitWidth 2..8} (small dictionaries, deeper level streams): a group is at most 64 bits,
     *       so one little-endian 64-bit load holds all 8 values.</li>
     *   <li>{@code bitWidth 9..32}: generic 64-bit accumulator loop.</li>
     * </ul>
     */
    public void readBitPackedGroups(int bitWidth, int groupCount8, int[] out, int offset, int maxWrite) {
        final long totalValues = (long) groupCount8 * 8L;
        final int toWrite = (int) Math.min(totalValues, Math.max(maxWrite, 0));
        if (bitWidth == 0) {
            java.util.Arrays.fill(out, offset, offset + toWrite, 0);
            return;
        }
        if (bitWidth < 0 || bitWidth > 32) {
            throw new IllegalArgumentException("invalid bit width " + bitWidth);
        }
        final long needBytes = (long) groupCount8 * bitWidth;
        if (pos < 0 || needBytes > (long) limit - pos) {
            throw new Corrupt(pos, (int) Math.min(needBytes, Integer.MAX_VALUE), limit, buf.length);
        }

        final byte[] b = buf;
        int p = pos;
        int o = offset;
        final int fullGroups = toWrite >>> 3;

        if (bitWidth == 1) {
            for (int g = 0; g < fullGroups; g++, o += 8) {
                final int x = b[p++] & 0xFF;
                out[o] = x & 1;
                out[o + 1] = (x >>> 1) & 1;
                out[o + 2] = (x >>> 2) & 1;
                out[o + 3] = (x >>> 3) & 1;
                out[o + 4] = (x >>> 4) & 1;
                out[o + 5] = (x >>> 5) & 1;
                out[o + 6] = (x >>> 6) & 1;
                out[o + 7] = (x >>> 7) & 1;
            }
        } else if (bitWidth <= 8) {
            final int mask = (1 << bitWidth) - 1;
            final int lastWideLoadStart = b.length - 8; // a 64-bit load at p needs p <= b.length - 8
            for (int g = 0; g < fullGroups; g++, o += 8, p += bitWidth) {
                if (p <= lastWideLoadStart) {
                    final long x = (long) LONG_LE.get(b, p);
                    out[o] = (int) x & mask;
                    out[o + 1] = (int) (x >>> bitWidth) & mask;
                    out[o + 2] = (int) (x >>> (2 * bitWidth)) & mask;
                    out[o + 3] = (int) (x >>> (3 * bitWidth)) & mask;
                    out[o + 4] = (int) (x >>> (4 * bitWidth)) & mask;
                    out[o + 5] = (int) (x >>> (5 * bitWidth)) & mask;
                    out[o + 6] = (int) (x >>> (6 * bitWidth)) & mask;
                    out[o + 7] = (int) (x >>> (7 * bitWidth)) & mask;
                } else {
                    unpack(b, p, bitWidth, out, o, 8); // last few groups of the buffer: byte-wise, never reads past b.length
                }
            }
        } else {
            for (int g = 0; g < fullGroups; g++, o += 8, p += bitWidth) {
                unpack(b, p, bitWidth, out, o, 8);
            }
        }

        final int tail = toWrite & 7;
        if (tail != 0) {
            // A partial group is only possible when maxWrite ends inside a run; decode just the values wanted.
            unpack(b, fullGroups * bitWidth + pos, bitWidth, out, o, tail);
        }
        pos += (int) needBytes; // skip the whole run, including any values beyond maxWrite
    }

    /** Unpacks {@code n} (<= 8) LSB-first {@code bitWidth}-bit values starting at {@code b[p]}. Caller has bounds-checked the run. */
    private static void unpack(byte[] b, int p, int bitWidth, int[] out, int o, int n) {
        final int mask = (bitWidth == 32) ? -1 : (1 << bitWidth) - 1;
        long acc = 0;
        int bits = 0;
        for (int i = 0; i < n; i++) {
            while (bits < bitWidth) {
                acc |= (long) (b[p++] & 0xFF) << bits;
                bits += 8;
            }
            out[o + i] = (int) acc & mask;
            acc >>>= bitWidth;
            bits -= bitWidth;
        }
    }
}
