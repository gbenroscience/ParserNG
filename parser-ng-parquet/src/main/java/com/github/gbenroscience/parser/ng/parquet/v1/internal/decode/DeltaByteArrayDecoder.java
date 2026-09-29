package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

/**
 * Decodes DELTA_LENGTH_BYTE_ARRAY (encoding id 6) and DELTA_BYTE_ARRAY (encoding id 7), Parquet's two
 * BINARY-only variable-length encodings, per the parquet-format spec (Encodings.md):
 *
 * <ul>
 *   <li><b>DELTA_LENGTH_BYTE_ARRAY</b>: every value's length, DELTA_BINARY_PACKED-encoded, followed by
 *       all the values' bytes concatenated back to back in the same order.</li>
 *   <li><b>DELTA_BYTE_ARRAY</b> (incremental / front-compression encoding): every value's shared-prefix
 *       length with the <em>previous</em> value, DELTA_BINARY_PACKED-encoded, followed by every value's
 *       suffix encoded as DELTA_LENGTH_BYTE_ARRAY (suffix lengths, then concatenated suffix bytes). Value
 *       {@code i} is reconstructed as {@code previous[0, prefixLen[i]) + suffix[i]}; value 0's "previous"
 *       is the empty string (a well-formed stream always has prefixLen[0] == 0 -- checked defensively
 *       below regardless, since this decodes untrusted input).</li>
 * </ul>
 *
 * <p>Only present entries are encoded (see {@link DeltaBinaryPackedDecoder}'s class Javadoc); {@code
 * count} here is always a page's present-entry count. Package-private: {@link MaterializedBinary}, the
 * output buffer, is itself package-private, owned and reused page-to-page by {@link FastColumnCursor}.
 */
final class DeltaByteArrayDecoder {

    private DeltaByteArrayDecoder() { }

    /** Decodes DELTA_LENGTH_BYTE_ARRAY into {@code out}, advancing {@code br} past lengths and bytes. */
    static void decodeLengthByteArray(ByteReader br, int count, MaterializedBinary out) {
        long[] lengths = new long[count];
        DeltaBinaryPackedDecoder.decode(br, lengths, count);
        out.reset(count);
        for (int i = 0; i < count; i++) {
            int len = checkedLen(lengths[i], br);
            out.append(i, br.buf, br.pos, len);
            br.pos += len;
        }
        out.finish(count);
    }

    /** Decodes DELTA_BYTE_ARRAY (prefix lengths, then suffix-as-DELTA_LENGTH_BYTE_ARRAY) into {@code out}. */
    static void decodeByteArray(ByteReader br, int count, MaterializedBinary out) {
        long[] prefixLen = new long[count];
        DeltaBinaryPackedDecoder.decode(br, prefixLen, count);
        long[] suffixLen = new long[count];
        DeltaBinaryPackedDecoder.decode(br, suffixLen, count);

        out.reset(count);
        int prevStart = 0, prevLen = 0;
        for (int i = 0; i < count; i++) {
            int plen = checkedLen(prefixLen[i], br);
            int slen = checkedLen(suffixLen[i], br);
            if (plen > prevLen) {
                throw new ByteReader.Corrupt(br.pos, plen, prevLen, br.buf.length); // prefix longer than the value it's a prefix of
            }
            out.appendSelf(i, prevStart, plen, br.buf, br.pos, slen);
            br.pos += slen;
            prevStart = out.offsets[i];
            prevLen = plen + slen;
        }
        out.finish(count);
    }

    private static int checkedLen(long len, ByteReader br) {
        if (len < 0 || len > Integer.MAX_VALUE) throw new ByteReader.Corrupt(br.pos, 0, br.limit, br.buf.length);
        return (int) len;
    }
}
