package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ByteStreamSplitDecoderTest {

    /** Independent reference encoder mirroring arrow-rs's {@code split_streams_const}: {@code dst[i + j*stride] = src[i*width+j]}. */
    private static byte[] encode(byte[] valuesLE, int width) {
        int count = valuesLE.length / width;
        byte[] out = new byte[valuesLE.length];
        for (int i = 0; i < count; i++)
            for (int j = 0; j < width; j++)
                out[i + j * count] = valuesLE[i * width + j];
        return out;
    }

    private static byte[] leInt(int v) { return new byte[]{(byte) v, (byte) (v >> 8), (byte) (v >> 16), (byte) (v >> 24)}; }

    private static byte[] leLong(long v) {
        byte[] b = new byte[8];
        for (int i = 0; i < 8; i++) b[i] = (byte) (v >>> (8 * i));
        return b;
    }

    @Test
    void int32RoundTrip() throws Exception {
        Random rnd = new Random(1);
        int n = 137;
        int[] vals = new int[n];
        ByteArrayOutputStream le = new ByteArrayOutputStream();
        for (int i = 0; i < n; i++) { vals[i] = rnd.nextInt(); le.write(leInt(vals[i])); }
        byte[] enc = encode(le.toByteArray(), 4);
        int[] out = new int[n];
        ByteStreamSplitDecoder.decodeInts(enc, 0, out, n);
        assertArrayEquals(vals, out);
    }

    @Test
    void int64RoundTrip() throws Exception {
        Random rnd = new Random(2);
        int n = 89;
        long[] vals = new long[n];
        ByteArrayOutputStream le = new ByteArrayOutputStream();
        for (int i = 0; i < n; i++) { vals[i] = rnd.nextLong(); le.write(leLong(vals[i])); }
        byte[] enc = encode(le.toByteArray(), 8);
        long[] out = new long[n];
        ByteStreamSplitDecoder.decodeLongs(enc, 0, out, n);
        assertArrayEquals(vals, out);
    }

    @Test
    void floatRoundTrip() throws Exception {
        Random rnd = new Random(3);
        int n = 51;
        float[] vals = new float[n];
        ByteArrayOutputStream le = new ByteArrayOutputStream();
        for (int i = 0; i < n; i++) { vals[i] = rnd.nextFloat() * 1000 - 500; le.write(leInt(Float.floatToIntBits(vals[i]))); }
        byte[] enc = encode(le.toByteArray(), 4);
        float[] out = new float[n];
        ByteStreamSplitDecoder.decodeFloats(enc, 0, out, n);
        assertArrayEquals(vals, out);
    }

    @Test
    void doubleRoundTrip() throws Exception {
        Random rnd = new Random(4);
        int n = 63;
        double[] vals = new double[n];
        ByteArrayOutputStream le = new ByteArrayOutputStream();
        for (int i = 0; i < n; i++) { vals[i] = rnd.nextDouble() * 1e10 - 5e9; le.write(leLong(Double.doubleToLongBits(vals[i]))); }
        byte[] enc = encode(le.toByteArray(), 8);
        double[] out = new double[n];
        ByteStreamSplitDecoder.decodeDoubles(enc, 0, out, n);
        assertArrayEquals(vals, out);
    }

    @Test
    void fixedLenByteArrayRoundTrip() {
        Random rnd = new Random(5);
        int n = 40, width = 5; // e.g. a FIXED_LEN_BYTE_ARRAY(5) DECIMAL(9,2) column
        byte[] flat = new byte[n * width];
        rnd.nextBytes(flat);
        byte[] enc = encode(flat, width);
        MaterializedBinary mb = new MaterializedBinary();
        ByteStreamSplitDecoder.decodeFixed(enc, 0, n, width, mb);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < width; j++) {
                assertEquals(flat[i * width + j], mb.data[mb.offsets[i] + j], "value " + i + " byte " + j);
            }
        }
    }

    @Test
    void respectsNonZeroBufferOffset() throws Exception {
        // A BYTE_STREAM_SPLIT-encoded page rarely starts at byte 0 of the underlying buffer.
        Random rnd = new Random(6);
        int n = 20;
        int[] vals = new int[n];
        ByteArrayOutputStream le = new ByteArrayOutputStream();
        for (int i = 0; i < n; i++) { vals[i] = rnd.nextInt(1000) - 500; le.write(leInt(vals[i])); }
        byte[] enc = encode(le.toByteArray(), 4);
        byte[] padded = new byte[7 + enc.length];
        System.arraycopy(enc, 0, padded, 7, enc.length);
        int[] out = new int[n];
        ByteStreamSplitDecoder.decodeInts(padded, 7, out, n);
        assertArrayEquals(vals, out);
    }

    @Test
    void rejectsTruncatedBuffer() {
        byte[] tooShort = new byte[10]; // needs 5*4=20 bytes for 5 int32 values
        int[] out = new int[5];
        org.junit.jupiter.api.Assertions.assertThrows(ByteReader.Corrupt.class,
                () -> ByteStreamSplitDecoder.decodeInts(tooShort, 0, out, 5));
    }
}
