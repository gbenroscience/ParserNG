package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

/**
 * Decodes BYTE_STREAM_SPLIT (encoding id 8): "K byte-streams are created where K is the size in bytes
 * of the data type. The individual bytes of a value are scattered to the corresponding stream and the
 * streams are concatenated" (parquet-format Encodings.md). Originally FLOAT/DOUBLE-only; PARQUET-2414
 * extended it to INT32, INT64 and FIXED_LEN_BYTE_ARRAY, which this decoder supports uniformly.
 *
 * <p>Layout, cross-checked against the reference Rust encoder (arrow-rs's {@code
 * byte_stream_split_encoder.rs}, {@code split_streams_const}: {@code dst[i + j*stride] = src[i*width +
 * j]} where {@code stride} is the value count): value {@code i}'s byte {@code j} (little-endian, as
 * every Parquet PLAIN fixed-width encoding is) lives at {@code encoded[j*count + i]} -- byte-stream
 * {@code j} occupies the contiguous run {@code [j*count, (j+1)*count)}. Every value's bytes are
 * scattered across all {@code width} streams, so -- like DELTA_BYTE_ARRAY -- none of this is zero-copy;
 * the fixed-width (INT32/INT64/FLOAT/DOUBLE) cases materialize into the same scratch arrays PLAIN's
 * bulk path fills (see {@code FastColumnCursor.setUpValueDecode}'s BYTE_STREAM_SPLIT branch), and
 * FIXED_LEN_BYTE_ARRAY materializes into a {@link MaterializedBinary}, exactly as DELTA_BYTE_ARRAY does.
 */
final class ByteStreamSplitDecoder {

    private ByteStreamSplitDecoder() { }

    static void decodeInts(byte[] buf, int off, int[] out, int count) {
        checkBounds(buf, off, count, 4);
        for (int i = 0; i < count; i++) {
            out[i] = (buf[off + i] & 0xFF)
                    | (buf[off + count + i] & 0xFF) << 8
                    | (buf[off + 2 * count + i] & 0xFF) << 16
                    | (buf[off + 3 * count + i] & 0xFF) << 24;
        }
    }

    static void decodeLongs(byte[] buf, int off, long[] out, int count) {
        checkBounds(buf, off, count, 8);
        for (int i = 0; i < count; i++) {
            long v = 0;
            for (int j = 0; j < 8; j++) {
                v |= (long) (buf[off + j * count + i] & 0xFF) << (8 * j);
            }
            out[i] = v;
        }
    }

    static void decodeFloats(byte[] buf, int off, float[] out, int count) {
        checkBounds(buf, off, count, 4);
        for (int i = 0; i < count; i++) {
            int bits = (buf[off + i] & 0xFF)
                    | (buf[off + count + i] & 0xFF) << 8
                    | (buf[off + 2 * count + i] & 0xFF) << 16
                    | (buf[off + 3 * count + i] & 0xFF) << 24;
            out[i] = Float.intBitsToFloat(bits);
        }
    }

    static void decodeDoubles(byte[] buf, int off, double[] out, int count) {
        checkBounds(buf, off, count, 8);
        for (int i = 0; i < count; i++) {
            long bits = 0;
            for (int j = 0; j < 8; j++) {
                bits |= (long) (buf[off + j * count + i] & 0xFF) << (8 * j);
            }
            out[i] = Double.longBitsToDouble(bits);
        }
    }

    /** {@code typeLength}-wide values, one entry per value in {@code out}, gathered from {@code typeLength} scattered streams. */
    static void decodeFixed(byte[] buf, int off, int count, int typeLength, MaterializedBinary out) {
        checkBounds(buf, off, count, typeLength);
        out.reset(count);
        byte[] scratch = new byte[typeLength];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < typeLength; j++) scratch[j] = buf[off + j * count + i];
            out.append(i, scratch, 0, typeLength);
        }
        out.finish(count);
    }

    private static void checkBounds(byte[] buf, int off, int count, int width) {
        long need = (long) count * width;
        if (off < 0 || need > (long) buf.length - off) {
            throw new ByteReader.Corrupt(off, (int) Math.min(need, Integer.MAX_VALUE), buf.length, buf.length);
        }
    }
}
