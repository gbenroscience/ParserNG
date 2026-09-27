package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

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
     * Reads {@code groupCount8 * 8} bit-packed values from the input, in order — but writes only
     * the first {@code maxWrite} of them into {@code out[offset..offset+maxWrite)}; any beyond that
     * are decoded (to keep {@link #pos} correctly advanced for whatever follows in the page) and
     * then discarded, never written.
     *
     * <p>This is a correctness requirement, not an optimization: {@code groupCount8} comes directly
     * from an unsigned varint in the file (the bit-packed run header) and is never validated against
     * the page's own remaining value count before this method runs. A well-formed encoder never
     * writes a run larger than needed to cover its remaining values, but the Parquet format does not
     * forbid it, and every Parquet file here is treated as untrusted input — correctness cannot
     * depend on encoders being well-behaved. A prior revision assumed the overshoot from a single
     * run was bounded to at most 7 slots (true only for a *minimally*-sized run) and pre-padded
     * output arrays to {@code count} rounded up to a multiple of 8 accordingly. That assumption
     * failed in production on a real file (count == 20000, an exact multiple of 8, giving zero
     * padding at all) and — more seriously — admits an unbounded-allocation path against a crafted
     * file declaring an oversized {@code groupCount8}: fuzzing 5,000 random streams that occasionally
     * over-allocate a run's group count found 1,205 failures under the old padding scheme, and zero
     * under this one. Bounding the write by the actual destination capacity, unconditionally, removes
     * both problems: there is no padding convention for a caller to get wrong, and no run size,
     * however large or maliciously chosen, can write outside {@code out} or force an allocation sized
     * by an attacker-controlled value.
     *
     * <p>Reading is bounds-checked the same way: a {@code groupCount8} large enough to demand more
     * input bytes than remain in this page's logical region throws {@link Corrupt} at the first byte
     * past {@link #limit}, instead of silently reading into whatever bytes happen to follow in the
     * backing array (the next column's page bytes, if this {@code buf} is shared/reused, or simply
     * garbage past the true page).
     */
    public void readBitPackedGroups(int bitWidth, int groupCount8, int[] out, int offset, int maxWrite) {
        int total = groupCount8 * 8;
        int toWrite = Math.min(total, Math.max(maxWrite, 0));
        if (bitWidth == 0) {
            java.util.Arrays.fill(out, offset, offset + toWrite, 0);
            return;
        }
        long bitBuffer = 0;
        int bitsInBuffer = 0, produced = 0;
        int mask = (bitWidth == 32) ? -1 : (1 << bitWidth) - 1;
        while (produced < total) {
            while (bitsInBuffer < bitWidth) {
                require(1);
                bitBuffer |= (long) (buf[pos++] & 0xFF) << bitsInBuffer;
                bitsInBuffer += 8;
            }
            int v = (int) (bitBuffer & mask);
            if (produced < toWrite) out[offset + produced] = v;
            produced++;
            bitBuffer >>>= bitWidth;
            bitsInBuffer -= bitWidth;
        }
    }
}
