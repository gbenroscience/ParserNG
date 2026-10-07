package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.DeltaTestUtil.*;
import static org.junit.jupiter.api.Assertions.*;

class DeltaBinaryPackedDecoderTest {

    // ------------------------------------------------------------ the spec's own worked examples,
    // ------------------------------------------------------------ transcribed verbatim (Encodings.md,
    // ------------------------------------------------------------ "Delta Encoding (DELTA_BINARY_PACKED = 5)")

    @Test
    void specExample1_constantDeltaBitwidthZero() {
        // "1, 2, 3, 4, 5" -- deltas are all 1, min delta 1, adjusted deltas all 0 -> bitwidth 0, no data bytes.
        byte[] raw = encodeDeltaBinaryPacked(8, 1, new long[]{1, 2, 3, 4, 5});
        ByteReader br = reader(raw);
        long[] out = new long[5];
        DeltaBinaryPackedDecoder.decode(br, out, 5);
        assertArrayEquals(new long[]{1, 2, 3, 4, 5}, out);
        assertEquals(raw.length, br.pos, "must consume exactly the delta-encoded section");
    }

    @Test
    void specExample2_negativeDeltasAndBitwidthTwo() {
        // "7, 5, 3, 1, 2, 3, 4, 5" -- deltas -2,-2,-2,1,1,1,1; min -2; adjusted 0,0,0,3,3,3,3 -> bitwidth 2.
        byte[] raw = encodeDeltaBinaryPacked(8, 1, new long[]{7, 5, 3, 1, 2, 3, 4, 5});
        ByteReader br = reader(raw);
        long[] out = new long[8];
        DeltaBinaryPackedDecoder.decode(br, out, 8);
        assertArrayEquals(new long[]{7, 5, 3, 1, 2, 3, 4, 5}, out);
        assertEquals(raw.length, br.pos);
    }

    @Test
    void singleValue_headerOnlyNoBlocks() {
        byte[] raw = encodeDeltaBinaryPacked(128, 4, new long[]{42});
        ByteReader br = reader(raw);
        long[] out = new long[1];
        DeltaBinaryPackedDecoder.decode(br, out, 1);
        assertArrayEquals(new long[]{42}, out);
        assertEquals(raw.length, br.pos);
    }

    @Test
    void multiBlockMultiMiniblockWithNegativeValuesAndPartialFinalBlock() {
        // 300 values with block size 128 / 4 miniblocks (32 values each): exercises multiple full blocks
        // plus a partially-filled final block, and a value range wide enough to need >32-bit deltas.
        long[] values = new long[300];
        Random rnd = new Random(42);
        long v = -1_000_000_000_000L;
        for (int i = 0; i < values.length; i++) {
            v += (rnd.nextLong() % 1_000_000_000L);
            values[i] = v;
        }
        byte[] raw = encodeDeltaBinaryPacked(128, 4, values);
        ByteReader br = reader(raw);
        long[] out = new long[values.length];
        DeltaBinaryPackedDecoder.decode(br, out, values.length);
        assertArrayEquals(values, out);
        assertEquals(raw.length, br.pos, "must fully consume every miniblock's padding, not stop early once `remaining` hits 0");
    }

    @Test
    void shortFinalBlockWithStaleWidthsFromPreviousBlock_asWrittenByParquetJava() {
        // Several lengths around the block boundaries, with varying deltas so every miniblock of every full block has a
        // non-zero width; the short last block then carries those stale widths in its unused slots, with no data behind them.
        Random rnd = new Random(7);
        for (int n : new int[]{2, 33, 129, 130, 161, 257, 258, 300, 301, 400, 1000}) {
            for (int[] shape : new int[][]{{128, 4}, {64, 2}, {256, 8}}) {
                long[] values = new long[n];
                long v = 1000;
                for (int i = 0; i < n; i++) { v += rnd.nextInt(40); values[i] = v; }
                byte[] raw = encodeDeltaBinaryPackedLikeParquetJava(shape[0], shape[1], values);
                ByteReader br = reader(raw);
                long[] out = new long[n];
                DeltaBinaryPackedDecoder.decode(br, out, n);
                assertArrayEquals(values, out, "n=" + n + " block=" + shape[0] + "/" + shape[1]);
                assertEquals(raw.length, br.pos, "n=" + n + " block=" + shape[0] + "/" + shape[1] + ": must end exactly where the section ends");
            }
        }
    }

    @Test
    void twoConsecutiveSequencesWithStaleWidths_theDeltaByteArrayShape() {
        // DELTA_BYTE_ARRAY = prefix lengths then suffix lengths, back to back. The second header is only found at the
        // right place if the first sequence consumed exactly its own bytes.
        Random rnd = new Random(11);
        for (int n : new int[]{130, 300, 513}) {
            long[] a = new long[n], b = new long[n];
            long x = 0, y = 0;
            for (int i = 0; i < n; i++) { x += rnd.nextInt(7); y += rnd.nextInt(25); a[i] = x; b[i] = y; }
            byte[] first = encodeDeltaBinaryPackedLikeParquetJava(128, 4, a), second = encodeDeltaBinaryPackedLikeParquetJava(128, 4, b);
            byte[] both = new byte[first.length + second.length];
            System.arraycopy(first, 0, both, 0, first.length);
            System.arraycopy(second, 0, both, first.length, second.length);
            ByteReader br = reader(both);
            long[] outA = new long[n], outB = new long[n];
            DeltaBinaryPackedDecoder.decode(br, outA, n);
            assertEquals(first.length, br.pos, "first sequence must stop exactly at the second header, n=" + n);
            DeltaBinaryPackedDecoder.decode(br, outB, n);
            assertArrayEquals(a, outA);
            assertArrayEquals(b, outB);
            assertEquals(both.length, br.pos);
        }
    }

    @Test
    void int32NarrowingViaDecodeInts() {
        long[] values = new long[]{100, 105, 90, 90, 500, -3000};
        byte[] raw = encodeDeltaBinaryPacked(8, 1, values);
        ByteReader br = reader(raw);
        long[] scratch = new long[values.length];
        int[] out = new int[values.length];
        DeltaBinaryPackedDecoder.decodeInts(br, scratch, out, values.length);
        for (int i = 0; i < values.length; i++) assertEquals((int) values[i], out[i]);
    }

    @Test
    void mismatchedDeclaredCountIsRejected() {
        byte[] raw = encodeDeltaBinaryPacked(8, 1, new long[]{1, 2, 3});
        ByteReader br = reader(raw);
        long[] out = new long[5];
        // The page's own present-count (from levels) disagrees with the stream's declared total_value_count.
        assertThrows(ByteReader.Corrupt.class, () -> DeltaBinaryPackedDecoder.decode(br, out, 5));
    }
}