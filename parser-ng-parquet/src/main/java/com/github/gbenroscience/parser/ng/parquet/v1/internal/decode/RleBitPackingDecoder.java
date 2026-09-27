package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

/**
 * Decodes the RLE/bit-packing hybrid stream parquet uses for definition/repetition levels and
 * {@code RLE_DICTIONARY} value indices. One call decodes a whole page's worth of values into a
 * caller-owned {@code int[]} — no per-value object, no per-value virtual dispatch (parquet-column's
 * {@code RunLengthBitPackingHybridDecoder} is value-at-a-time through a {@code ValuesReader}).
 */
public final class RleBitPackingDecoder {

    private RleBitPackingDecoder() {}

    /**
     * Decodes exactly {@code count} logical values into {@code out[0..count)}.
     *
     * <p><b>Caller contract:</b> {@code out.length} must be at least {@code count} — nothing more.
     * A bit-packed run's declared size can legally exceed what remains to be produced (see
     * {@link ByteReader#readBitPackedGroups}); this method never writes past {@code out.length}
     * regardless, by construction, not by any padding convention the caller has to get right.
     */
    public static void decode(ByteReader br, int bitWidth, int[] out, int count) {
        if (bitWidth == 0) {
            java.util.Arrays.fill(out, 0, count, 0);
            return;
        }
        int valueByteWidth = (bitWidth + 7) >>> 3;
        int produced = 0;
        while (produced < count) {
            int header = br.readUnsignedVarInt();
            if ((header & 1) == 0) {
                int runLength = header >>> 1;
                int value = readLittleEndianValue(br, valueByteWidth);
                int n = Math.min(runLength, count - produced);
                java.util.Arrays.fill(out, produced, produced + n, value);
                produced += n;
            } else {
                int groupCount8 = header >>> 1;
                br.readBitPackedGroups(bitWidth, groupCount8, out, produced, count - produced);
                produced += Math.min(groupCount8 * 8, count - produced);
            }
        }
    }

    private static int readLittleEndianValue(ByteReader br, int byteWidth) {
        int v = 0;
        for (int i = 0; i < byteWidth; i++) {
            v |= (br.readByte() & 0xFF) << (8 * i);
        }
        return v;
    }
}