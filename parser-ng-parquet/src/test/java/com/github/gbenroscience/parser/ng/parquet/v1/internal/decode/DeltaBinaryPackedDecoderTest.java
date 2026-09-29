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
