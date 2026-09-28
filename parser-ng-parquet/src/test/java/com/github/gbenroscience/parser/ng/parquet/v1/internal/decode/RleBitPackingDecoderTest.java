package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.Random;

import static com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.BitPackTestUtil.*;
import static org.junit.jupiter.api.Assertions.*;

class RleBitPackingDecoderTest {

    private static int[] decode(byte[] stream, int bitWidth, int count) {
        int[] out = new int[count];
        RleBitPackingDecoder.decode(reader(stream), bitWidth, out, count);
        return out;
    }

    @Test
    void singleRleRun() {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        rleRun(o, 10, 3, 2);
        int[] expected = new int[10];
        Arrays.fill(expected, 3);
        assertArrayEquals(expected, decode(o.toByteArray(), 2, 10));
    }

    @Test
    void singleBitPackedRun() {
        int[] values = {0, 1, 2, 3, 3, 2, 1, 0};
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        bitPackedRun(o, values, 2);
        assertArrayEquals(values, decode(o.toByteArray(), 2, 8));
    }

    @Test
    void mixedRunsAreStitchedInOrder() {
        int[] packed = {5, 6, 7, 1, 2, 3, 4, 0};
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        rleRun(o, 3, 7, 3);
        bitPackedRun(o, packed, 3);
        rleRun(o, 2, 1, 3);
        int[] expected = {7, 7, 7, 5, 6, 7, 1, 2, 3, 4, 0, 1, 1};
        assertArrayEquals(expected, decode(o.toByteArray(), 3, expected.length));
    }

    @Test
    void countSmallerThanTheRunTruncatesIt() {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        rleRun(o, 100, 9, 4);
        assertArrayEquals(new int[]{9, 9, 9, 9, 9}, decode(o.toByteArray(), 4, 5));
    }

    @Test
    void paddedBitPackedTailNeverWritesPastTheOutputArray() {
        // One group declares 8 values; only 3 are wanted and out.length is exactly 3.
        int[] values = {1, 2, 3, 0, 0, 0, 0, 0};
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        bitPackedRun(o, values, 2);
        assertArrayEquals(new int[]{1, 2, 3}, decode(o.toByteArray(), 2, 3));
    }

    @Test
    void multiByteRleValuesAcrossBitWidthBands() {
        int[][] cases = {{9, 0x1FF}, {13, 0x1234}, {16, 0xABCD}, {17, 0x1ABCD}, {24, 0xABCDEF}, {25, 0x1ABCDEF}, {31, 0x7ABCDEF1}};
        for (int[] c : cases) {
            ByteArrayOutputStream o = new ByteArrayOutputStream();
            rleRun(o, 4, c[1], c[0]);
            assertArrayEquals(new int[]{c[1], c[1], c[1], c[1]}, decode(o.toByteArray(), c[0], 4), "bitWidth=" + c[0]);
        }
    }

    @Test
    void bitWidthZeroNeedsNoBytes() {
        int[] out = {9, 9, 9, 9};
        ByteReader br = reader(new byte[0]);
        RleBitPackingDecoder.decode(br, 0, out, 3);
        assertArrayEquals(new int[]{0, 0, 0, 9}, out);
    }

    @Test
    void nonPositiveCountReadsNothing() {
        ByteReader br = reader(new byte[]{1, 2, 3});
        RleBitPackingDecoder.decode(br, 4, new int[0], 0);
        RleBitPackingDecoder.decode(br, 4, new int[0], -5);
        assertEquals(3, br.remaining());
    }

    @Test
    void invalidBitWidthsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> RleBitPackingDecoder.decode(reader(new byte[8]), 33, new int[1], 1));
        assertThrows(IllegalArgumentException.class, () -> RleBitPackingDecoder.decode(reader(new byte[8]), -1, new int[1], 1));
    }

    @Test
    void truncatedStreamIsCorrupt() {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        rleRun(o, 4, 5, 8);
        byte[] cut = Arrays.copyOf(o.toByteArray(), o.size() - 1); // drop the value byte
        assertThrows(ByteReader.Corrupt.class, () -> decode(cut, 8, 4));
        assertThrows(ByteReader.Corrupt.class, () -> decode(new byte[0], 8, 1));
    }

    @Test
    void hostileRleRunLengthIsClampedToTheRequestedCount() {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        varint(o, 0xFFFFFFFEL); // even header -> RLE run of 0x7FFFFFFF values
        o.write(1);
        assertArrayEquals(new int[]{1, 1, 1, 1, 1}, decode(o.toByteArray(), 1, 5));
    }

    @Test
    void hostileBitPackedGroupCountIsCorruptRatherThanAnAllocationOrOverflow() {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        varint(o, 0xFFFFFFFFL); // odd header -> 0x7FFFFFFF groups
        o.write(0xAA);
        assertThrows(ByteReader.Corrupt.class, () -> decode(o.toByteArray(), 8, 4));
    }

    @Test
    void randomisedHybridStreamsRoundTrip() {
        Random rnd = new Random(20260928L);
        for (int iter = 0; iter < 300; iter++) {
            int bw = 1 + rnd.nextInt(32);
            ByteArrayOutputStream o = new ByteArrayOutputStream();
            int[] expected = new int[0];
            int runs = 1 + rnd.nextInt(8);
            for (int r = 0; r < runs; r++) {
                int[] chunk;
                if (rnd.nextBoolean()) {
                    int len = 1 + rnd.nextInt(40);
                    int v = rnd.nextInt() & mask(bw);
                    rleRun(o, len, v, bw);
                    chunk = new int[len];
                    Arrays.fill(chunk, v);
                } else {
                    chunk = new int[8 * (1 + rnd.nextInt(5))];
                    for (int i = 0; i < chunk.length; i++) chunk[i] = rnd.nextInt() & mask(bw);
                    bitPackedRun(o, chunk, bw);
                }
                int[] merged = Arrays.copyOf(expected, expected.length + chunk.length);
                System.arraycopy(chunk, 0, merged, expected.length, chunk.length);
                expected = merged;
            }
            assertArrayEquals(expected, decode(o.toByteArray(), bw, expected.length), "iter=" + iter + " bw=" + bw);
        }
    }
}