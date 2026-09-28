package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;

import static com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.BitPackTestUtil.*;
import static org.junit.jupiter.api.Assertions.*;

class ByteReaderTest {

    // ------------------------------------------------------------ fixed-width reads

    @Test
    void littleEndianPrimitivesRoundTrip() {
        ByteBuffer bb = ByteBuffer.allocate(4 + 8 + 4 + 8).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(0x01020304).putLong(0x1122334455667788L).putFloat(3.5f).putDouble(-2.25d);
        ByteReader br = reader(bb.array());
        assertEquals(0x01020304, br.readIntLE());
        assertEquals(0x1122334455667788L, br.readLongLE());
        assertEquals(3.5f, br.readFloatLE());
        assertEquals(-2.25d, br.readDoubleLE());
        assertFalse(br.hasRemaining());
        assertEquals(0, br.remaining());
    }

    @Test
    void negativeValuesAreSignExtendedCorrectly() {
        ByteBuffer bb = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(-1).putLong(Long.MIN_VALUE);
        ByteReader br = reader(bb.array());
        assertEquals(-1, br.readIntLE());
        assertEquals(Long.MIN_VALUE, br.readLongLE());
    }

    @Test
    void readByteReturnsSignedBytes() {
        ByteReader br = reader(new byte[]{(byte) 0xFF, 0x7F});
        assertEquals((byte) 0xFF, br.readByte());
        assertEquals(0x7F, br.readByte());
    }

    @Test
    void wrapHonoursOffsetAndLength() {
        byte[] buf = {9, 9, 1, 0, 0, 0, 9, 9};
        ByteReader br = new ByteReader();
        br.wrap(buf, 2, 4);
        assertEquals(4, br.remaining());
        assertEquals(1, br.readIntLE());
        assertFalse(br.hasRemaining());
    }

    @Test
    void rewrapResetsTheCursor() {
        ByteReader br = reader(new byte[]{1, 2});
        br.readByte();
        br.wrap(new byte[]{5, 6, 7}, 1, 2);
        assertEquals(2, br.remaining());
        assertEquals(6, br.readByte());
    }

    // ------------------------------------------------------------ bounds

    @Test
    void readingPastTheEndThrowsCorruptWithCursorContext() {
        ByteReader br = reader(new byte[]{1, 2, 3});
        ByteReader.Corrupt e = assertThrows(ByteReader.Corrupt.class, br::readIntLE);
        assertEquals(0, e.pos);
        assertEquals(4, e.need);
        assertEquals(3, e.limit);
        assertEquals(3, e.capacity);
        assertTrue(e.getMessage().contains("underrun"), e.getMessage());
    }

    @Test
    void logicalLimitIsEnforcedEvenWhenPhysicalBytesRemain() {
        byte[] buf = new byte[16];
        ByteReader br = new ByteReader();
        br.wrap(buf, 0, 4);
        br.readIntLE();
        assertThrows(ByteReader.Corrupt.class, br::readByte, "buffer has room, but the page ended");
    }

    @Test
    void aFailedReadDoesNotAdvanceTheCursor() {
        ByteReader br = reader(new byte[]{1, 2, 3});
        assertThrows(ByteReader.Corrupt.class, br::readIntLE);
        assertEquals(3, br.remaining());
    }

    @Test
    void longReadThatStraddlesTheLimitFails() {
        ByteReader br = reader(new byte[6]);
        assertThrows(ByteReader.Corrupt.class, br::readLongLE);
    }

    // ------------------------------------------------------------ varint

    private static int varintOf(int... bytes) {
        byte[] b = new byte[bytes.length];
        for (int i = 0; i < b.length; i++) b[i] = (byte) bytes[i];
        return reader(b).readUnsignedVarInt();
    }

    @Test
    void varintDecodesKnownEncodings() {
        assertEquals(0, varintOf(0x00));
        assertEquals(1, varintOf(0x01));
        assertEquals(127, varintOf(0x7F));
        assertEquals(128, varintOf(0x80, 0x01));
        assertEquals(300, varintOf(0xAC, 0x02));
        assertEquals(16384, varintOf(0x80, 0x80, 0x01));
    }

    @Test
    void varintRoundTripsAgainstTheReferenceEncoder() {
        Random rnd = new Random(7);
        for (int i = 0; i < 500; i++) {
            // A random value of 1..32 significant bits, so every varint length (1..5 bytes) is exercised.
            // (The previous rnd.nextInt(1 << k) threw "bound must be positive" whenever k reached 31, because
            // 1 << 31 is Integer.MIN_VALUE.) The reader returns the int bit pattern of an unsigned 32-bit
            // value, and the reference encoder takes a long, so widen WITHOUT sign extension.
            int bits = 1 + rnd.nextInt(32);
            int v = (int) (rnd.nextLong() & ((1L << bits) - 1));
            ByteArrayOutputStream o = new ByteArrayOutputStream();
            varint(o, Integer.toUnsignedLong(v));
            ByteReader br = reader(o.toByteArray());
            assertEquals(v, br.readUnsignedVarInt());
            assertFalse(br.hasRemaining());
        }
    }

    @Test
    void fiveByteVarintCoversTheFull32BitRange() {
        assertEquals(-1, varintOf(0xFF, 0xFF, 0xFF, 0xFF, 0x0F), "0xFFFFFFFF is returned as its int bit pattern");
    }

    @Test
    void varintWithoutATerminatorIsCorrupt() {
        assertThrows(ByteReader.Corrupt.class, () -> varintOf(0x80));
        assertThrows(ByteReader.Corrupt.class, () -> varintOf(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00),
                "more continuation bytes than any 32-bit value needs is malformed, not just long");
    }

    // ------------------------------------------------------------ bit-packed groups

    @Test
    void everyBitWidthUnpacksCorrectly() {
        Random rnd = new Random(42);
        for (int bw = 1; bw <= 32; bw++) {
            int[] values = new int[24]; // 3 groups
            for (int i = 0; i < values.length; i++) values[i] = rnd.nextInt() & mask(bw);
            ByteReader br = reader(pack(values, bw));
            int[] out = new int[values.length];
            br.readBitPackedGroups(bw, 3, out, 0, out.length);
            assertArrayEquals(values, out, "bitWidth=" + bw);
            assertFalse(br.hasRemaining(), "bitWidth=" + bw + " must consume exactly groups*bitWidth bytes");
        }
    }

    @Test
    void allOnesAtEveryBitWidth() {
        for (int bw = 1; bw <= 32; bw++) {
            int[] values = new int[16];
            Arrays.fill(values, mask(bw));
            int[] out = new int[16];
            reader(pack(values, bw)).readBitPackedGroups(bw, 2, out, 0, 16);
            assertArrayEquals(values, out, "bitWidth=" + bw);
        }
    }

    @Test
    void maxWriteLimitsWritesButStillConsumesTheWholeRun() {
        int bw = 5;
        int[] values = new int[16];
        for (int i = 0; i < 16; i++) values[i] = i + 1;
        ByteReader br = reader(concat(pack(values, bw), new byte[]{0x55}));
        int[] out = new int[16];
        Arrays.fill(out, -1);
        br.readBitPackedGroups(bw, 2, out, 0, 11);
        for (int i = 0; i < 11; i++) assertEquals(values[i], out[i], "i=" + i);
        for (int i = 11; i < 16; i++) assertEquals(-1, out[i], "must not write beyond maxWrite, i=" + i);
        assertEquals(0x55, br.readByte(), "the reader must land exactly after the run");
    }

    @Test
    void offsetShiftsWherePlainValuesAreWritten() {
        int[] values = {1, 2, 3, 4, 5, 6, 7, 0};
        int[] out = new int[12];
        Arrays.fill(out, -1);
        reader(pack(values, 3)).readBitPackedGroups(3, 1, out, 2, 8);
        assertEquals(-1, out[0]);
        assertEquals(-1, out[1]);
        assertArrayEquals(values, Arrays.copyOfRange(out, 2, 10));
        assertEquals(-1, out[10]);
    }

    @Test
    void zeroMaxWriteWritesNothing() {
        int[] out = {-1, -1};
        ByteReader br = reader(pack(new int[8], 4));
        br.readBitPackedGroups(4, 1, out, 0, 0);
        assertArrayEquals(new int[]{-1, -1}, out);
        assertFalse(br.hasRemaining());
    }

    @Test
    void bitWidthZeroFillsWithZerosAndConsumesNothing() {
        int[] out = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
        ByteReader br = reader(new byte[]{1, 2});
        // One group is 8 values, but maxWrite=4 caps the writes at the first 4 slots: those become 0, the rest are untouched.
        br.readBitPackedGroups(0, 1, out, 0, 4);
        assertArrayEquals(new int[]{0, 0, 0, 0, 7, 7, 7, 7, 7, 7}, out);
        assertEquals(2, br.remaining(), "bit width 0 occupies no input bytes");

        // Uncapped: the whole group of 8 is zero-filled, still without consuming anything; offset shifts the fill.
        br.readBitPackedGroups(0, 1, out, 2, 100);
        assertArrayEquals(new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, out);
        assertEquals(2, br.remaining());
    }

    @Test
    void invalidBitWidthIsRejected() {
        ByteReader br = reader(new byte[64]);
        assertThrows(IllegalArgumentException.class, () -> br.readBitPackedGroups(33, 1, new int[8], 0, 8));
        assertThrows(IllegalArgumentException.class, () -> br.readBitPackedGroups(-1, 1, new int[8], 0, 8));
    }

    @Test
    void truncatedRunIsCorruptAndConsumesNothing() {
        ByteReader br = reader(new byte[3]); // 2 groups of 4-bit values need 8 bytes
        assertThrows(ByteReader.Corrupt.class, () -> br.readBitPackedGroups(4, 2, new int[16], 0, 16));
        assertEquals(3, br.remaining(), "checked up front, before any byte is consumed");
    }

    @Test
    void hostileGroupCountCannotOverflowOrAllocate() {
        ByteReader br = reader(new byte[4]);
        int[] out = new int[4];
        assertThrows(ByteReader.Corrupt.class,
                () -> br.readBitPackedGroups(32, Integer.MAX_VALUE, out, 0, out.length));
    }

    private static byte[] concat(byte[] a, byte[] b) {
        byte[] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }
}