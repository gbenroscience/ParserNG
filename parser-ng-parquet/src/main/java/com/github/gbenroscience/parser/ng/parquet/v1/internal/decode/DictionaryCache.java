package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScanException;
import org.apache.parquet.column.page.DictionaryPage;
import org.apache.parquet.schema.PrimitiveType;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Decodes one column chunk's {@link DictionaryPage} exactly once (dictionary pages are always
 * {@code PLAIN}-encoded regardless of whether data pages use {@code PLAIN_DICTIONARY} or
 * {@code RLE_DICTIONARY}) into flat arrays: a primitive array for fixed-width physical types, or a
 * single concatenated {@code byte[]} plus an offset table for {@code BINARY}. Gathering a
 * dictionary-encoded value during decode is then an array index, never a fresh {@code Binary}
 * object per row — the one-time dictionary decode below is amortized over every row that hits it.
 */
public final class DictionaryCache {

    private final PrimitiveType.PrimitiveTypeName physical;
    private int size;

    private int[] ints;
    private long[] longs;
    private float[] floats;
    private double[] doubles;
    private byte[] binData;
    private int[] binOffsets; // length size+1

    public DictionaryCache(DictionaryPage page, PrimitiveType.PrimitiveTypeName physical, Path file, String column, int typeLength) {
        this.physical = physical;
        try {
            byte[] raw = page.getBytes().toByteArray();
            int n = page.getDictionarySize();
            this.size = n;
            ByteReader br = new ByteReader();
            br.wrap(raw, 0, raw.length);
            switch (physical) {
                case INT32 -> {
                    ints = new int[n];
                    for (int i = 0; i < n; i++) ints[i] = br.readIntLE();
                }
                case INT64 -> {
                    longs = new long[n];
                    for (int i = 0; i < n; i++) longs[i] = br.readLongLE();
                }
                case FLOAT -> {
                    floats = new float[n];
                    for (int i = 0; i < n; i++) floats[i] = br.readFloatLE();
                }
                case DOUBLE -> {
                    doubles = new double[n];
                    for (int i = 0; i < n; i++) doubles[i] = br.readDoubleLE();
                }
                case BINARY -> {
                    binOffsets = new int[n + 1];
                    binData = new byte[raw.length - 4 * n];
                    int dataPos = 0;
                    for (int i = 0; i < n; i++) {
                        int len = br.readIntLE();
                        System.arraycopy(raw, br.pos, binData, dataPos, len);
                        br.pos += len;
                        binOffsets[i] = dataPos;
                        dataPos += len;
                    }
                    binOffsets[n] = dataPos;
                }
                case FIXED_LEN_BYTE_ARRAY -> {
                    // PLAIN dictionary entries for FIXED_LEN_BYTE_ARRAY carry no length prefix (unlike
                    // BINARY): every entry is exactly typeLength bytes, back to back.
                    binOffsets = new int[n + 1];
                    binData = new byte[n * typeLength];
                    for (int i = 0; i < n; i++) {
                        System.arraycopy(raw, br.pos, binData, i * typeLength, typeLength);
                        br.pos += typeLength;
                        binOffsets[i] = i * typeLength;
                    }
                    binOffsets[n] = n * typeLength;
                }
                case INT96 -> {
                    // See Int96Timestamp's Javadoc: converted to epoch nanoseconds once here, so
                    // FastColumnCursor.gatherDictionary's INT96 case is a plain array read, exactly like
                    // every other dictionary-encoded fixed-width type.
                    longs = new long[n];
                    for (int i = 0; i < n; i++) {
                        long nanosOfDay = br.readLongLE();
                        int julianDay = br.readIntLE();
                        longs[i] = Int96Timestamp.toEpochNanos(nanosOfDay, julianDay);
                    }
                }
                default -> throw new ParquetScanException("Dictionary encoding not supported for " + physical, file, -1, column, null);
            }
        } catch (IOException e) {
            throw new ParquetScanException("Failed to decode dictionary page", file, -1, column, e);
        }
    }

    public int size() { return size; }

    public int getInt(int idx) { return ints[idx]; }
    public long getLong(int idx) { return longs[idx]; }
    public float getFloat(int idx) { return floats[idx]; }
    public double getDouble(int idx) { return doubles[idx]; }

    /**
     * Bulk gathers: {@code out[i] = dictionary[idx[from + i]]} for {@code i in [0, n)}. A tight typed loop with no
     * per-value dispatch. An out-of-range index (corrupt page) surfaces as {@link ArrayIndexOutOfBoundsException};
     * the caller ({@link FastColumnCursor}) converts that into a {@code ParquetScanException} with file/column context.
     */
    public void gatherInts(int[] idx, int from, int[] out, int n) {
        final int[] d = ints;
        for (int i = 0; i < n; i++) out[i] = d[idx[from + i]];
    }

    public void gatherLongs(int[] idx, int from, long[] out, int n) {
        final long[] d = longs;
        for (int i = 0; i < n; i++) out[i] = d[idx[from + i]];
    }

    public void gatherFloats(int[] idx, int from, float[] out, int n) {
        final float[] d = floats;
        for (int i = 0; i < n; i++) out[i] = d[idx[from + i]];
    }

    public void gatherDoubles(int[] idx, int from, double[] out, int n) {
        final double[] d = doubles;
        for (int i = 0; i < n; i++) out[i] = d[idx[from + i]];
    }

    public byte[] binData() { return binData; }
    public int binOffset(int idx) { return binOffsets[idx]; }
    public int binLength(int idx) { return binOffsets[idx + 1] - binOffsets[idx]; }
}
