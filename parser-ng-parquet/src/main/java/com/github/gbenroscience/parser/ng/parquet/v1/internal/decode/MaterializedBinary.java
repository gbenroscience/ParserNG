package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import java.util.Arrays;

/**
 * Growable concatenated-bytes-plus-offsets buffer, one per column cursor, reused page to page.
 *
 * <p>Every other BINARY decode path in {@link FastColumnCursor} (PLAIN, dictionary) is zero-copy: it
 * points {@code curBinData}/{@code curBinOff}/{@code curBinLen} directly into bytes that already exist
 * (the page buffer, or {@link DictionaryCache}). DELTA_LENGTH_BYTE_ARRAY and, especially,
 * DELTA_BYTE_ARRAY cannot be: a DELTA_BYTE_ARRAY value is reconstructed from a prefix of the
 * <em>previous</em> value plus a new suffix, so it has no contiguous home in the original page bytes.
 * This class is that home instead -- one entry per present value in a page, offsets{@code [i]} to
 * offsets{@code [i+1]} bounding entry {@code i}'s bytes in {@code data}.
 */
final class MaterializedBinary {

    byte[] data = new byte[256];
    int[] offsets = new int[64]; // length present+1 after reset(); offsets[present] == dataLen
    int dataLen;

    /** Clears prior content and ensures {@link #offsets} can hold {@code presentCount + 1} entries. */
    void reset(int presentCount) {
        dataLen = 0;
        if (offsets.length < presentCount + 1) {
            int cap = offsets.length;
            while (cap < presentCount + 1) cap *= 2;
            offsets = new int[cap];
        }
    }

    /** Appends one value's bytes as entry {@code index}; caller calls in increasing index order, then {@link #finish}. */
    void append(int index, byte[] src, int srcOff, int len) {
        ensureData(dataLen + len);
        System.arraycopy(src, srcOff, data, dataLen, len);
        offsets[index] = dataLen;
        dataLen += len;
    }

    /** Same as {@link #append}, but the source is this buffer's own earlier region (DELTA_BYTE_ARRAY's prefix reuse). */
    void appendSelf(int index, int selfSrcOff, int prefixLen, byte[] suffixSrc, int suffixOff, int suffixLen) {
        final int total = prefixLen + suffixLen;
        ensureData(dataLen + total);
        if (prefixLen > 0) System.arraycopy(data, selfSrcOff, data, dataLen, prefixLen); // arraycopy tolerates overlap
        if (suffixLen > 0) System.arraycopy(suffixSrc, suffixOff, data, dataLen + prefixLen, suffixLen);
        offsets[index] = dataLen;
        dataLen += total;
    }

    /** Call once after the last {@link #append}/{@link #appendSelf}, with the total present-entry count. */
    void finish(int presentCount) {
        offsets[presentCount] = dataLen;
    }

    private void ensureData(int need) {
        if (data.length < need) {
            int cap = data.length;
            while (cap < need) cap *= 2;
            data = Arrays.copyOf(data, cap);
        }
    }
}
