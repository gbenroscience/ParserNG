package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import java.io.ByteArrayOutputStream;

/** Test-side reference encoder for Parquet's RLE/bit-packing hybrid (independent of the code under test). */
final class BitPackTestUtil {

    private BitPackTestUtil() { }

    /** LSB-first bit packing, exactly as Parquet lays it out. */
    static byte[] pack(int[] values, int bitWidth) {
        byte[] out = new byte[(int) (((long) values.length * bitWidth + 7) / 8)];
        long bitPos = 0;
        for (int v : values) {
            long x = v & 0xFFFFFFFFL;
            for (int b = 0; b < bitWidth; b++, bitPos++) {
                if (((x >>> b) & 1L) != 0) out[(int) (bitPos >>> 3)] |= (byte) (1 << (bitPos & 7));
            }
        }
        return out;
    }

    static void varint(ByteArrayOutputStream o, long v) {
        while ((v & ~0x7FL) != 0) {
            o.write((int) ((v & 0x7F) | 0x80));
            v >>>= 7;
        }
        o.write((int) v);
    }

    /** An RLE run: header (count << 1), then the value in ceil(bitWidth/8) little-endian bytes. */
    static void rleRun(ByteArrayOutputStream o, long count, int value, int bitWidth) {
        varint(o, count << 1);
        for (int i = 0; i < (bitWidth + 7) / 8; i++) o.write((value >>> (8 * i)) & 0xFF);
    }

    /** A bit-packed run; values.length must be a multiple of 8. */
    static void bitPackedRun(ByteArrayOutputStream o, int[] values, int bitWidth) {
        if (values.length % 8 != 0) throw new IllegalArgumentException("multiple of 8 required");
        varint(o, ((long) (values.length / 8) << 1) | 1L);
        byte[] packed = pack(values, bitWidth);
        o.write(packed, 0, packed.length);
    }

    static int mask(int bitWidth) {
        return bitWidth >= 32 ? -1 : (1 << bitWidth) - 1;
    }

    static ByteReader reader(byte[] bytes) {
        ByteReader br = new ByteReader();
        br.wrap(bytes, 0, bytes.length);
        return br;
    }
} 