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

    public DictionaryCache(DictionaryPage page, PrimitiveType.PrimitiveTypeName physical, Path file, String column) {
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

    public byte[] binData() { return binData; }
    public int binOffset(int idx) { return binOffsets[idx]; }
    public int binLength(int idx) { return binOffsets[idx + 1] - binOffsets[idx]; }
}
