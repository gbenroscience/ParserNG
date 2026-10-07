package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import java.io.ByteArrayOutputStream;
import java.util.Arrays;

/**
 * Test-side reference encoder for DELTA_BINARY_PACKED / DELTA_LENGTH_BYTE_ARRAY / DELTA_BYTE_ARRAY,
 * independent of the code under test -- mirrors {@code BitPackTestUtil}'s role for the RLE/bit-packing
 * hybrid. Implements the parquet-format spec's own encoding recipe (Encodings.md, "Delta Encoding")
 * directly, so a bug shared between this encoder and the decoder under test would have to reproduce the
 * same misreading of the spec in two independently-written implementations.
 */
final class DeltaTestUtil {

    private DeltaTestUtil() { }

    static void writeUVarInt(ByteArrayOutputStream out, long v) {
        while (true) {
            int b = (int) (v & 0x7F);
            v >>>= 7;
            if (v != 0) out.write(b | 0x80);
            else { out.write(b); break; }
        }
    }

    static void writeZigZag(ByteArrayOutputStream out, long v) {
        writeUVarInt(out, (v << 1) ^ (v >> 63));
    }

    /** LSB-first bit packing of exactly {@code values.length} values (a multiple of 8) at {@code bitWidth}. */
    static void writeBitPacked(ByteArrayOutputStream out, int bitWidth, long[] values) {
        if (bitWidth == 0) return;
        long acc = 0;
        int bits = 0;
        for (long v : values) {
            acc |= (v << bits);
            bits += bitWidth;
            while (bits >= 8) {
                out.write((int) (acc & 0xFF));
                acc >>>= 8;
                bits -= 8;
            }
        }
        if (bits > 0) out.write((int) (acc & 0xFF));
    }

    /**
     * Encodes {@code values} as DELTA_BINARY_PACKED with the given block size and miniblock count
     * (caller picks values that divide evenly, as any real writer's parameters would).
     */
    static byte[] encodeDeltaBinaryPacked(int blockSize, int miniblocksPerBlock, long[] values) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        writeUVarInt(out, blockSize);
        writeUVarInt(out, miniblocksPerBlock);
        writeUVarInt(out, values.length);
        writeZigZag(out, values[0]);
        int valuesPerMiniblock = blockSize / miniblocksPerBlock;
        int i = 1;
        while (i < values.length) {
            int blockCount = Math.min(blockSize, values.length - i);
            long[] deltas = new long[blockSize]; // padded with 0 beyond blockCount, per spec
            long prev = values[i - 1];
            for (int j = 0; j < blockCount; j++) {
                deltas[j] = values[i + j] - prev;
                prev = values[i + j];
            }
            long min = 0;
            if (blockCount > 0) {
                min = Long.MAX_VALUE;
                for (int j = 0; j < blockCount; j++) min = Math.min(min, deltas[j]);
            }
            long[] adjusted = new long[blockSize];
            for (int j = 0; j < blockSize; j++) adjusted[j] = (j < blockCount) ? (deltas[j] - min) : 0;
            writeZigZag(out, min);
            int[] widths = new int[miniblocksPerBlock];
            for (int m = 0; m < miniblocksPerBlock; m++) {
                long maxV = 0;
                for (int j = 0; j < valuesPerMiniblock; j++) maxV = Math.max(maxV, adjusted[m * valuesPerMiniblock + j]);
                widths[m] = 64 - Long.numberOfLeadingZeros(maxV);
                out.write(widths[m]);
            }
            for (int m = 0; m < miniblocksPerBlock; m++) {
                writeBitPacked(out, widths[m], Arrays.copyOfRange(adjusted, m * valuesPerMiniblock, (m + 1) * valuesPerMiniblock));
            }
            i += blockCount;
        }
        return out.toByteArray();
    }

    /**
     * DELTA_BINARY_PACKED laid out the way parquet-java's writer does it: one bit-width array that persists across
     * blocks and is only refreshed for the miniblocks a block actually uses, all {@code miniblocksPerBlock} widths
     * written for every block (so the unused slots of a short final block keep the previous block's, usually
     * non-zero, widths), but data written only for the miniblocks that hold values. {@link #encodeDeltaBinaryPacked}
     * pads with zero widths instead, which is spec-legal but is not what real files contain.
     */
    static byte[] encodeDeltaBinaryPackedLikeParquetJava(int blockSize, int miniblocksPerBlock, long[] values) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        writeUVarInt(out, blockSize);
        writeUVarInt(out, miniblocksPerBlock);
        writeUVarInt(out, values.length);
        writeZigZag(out, values[0]);
        int per = blockSize / miniblocksPerBlock;
        int[] widths = new int[miniblocksPerBlock];
        int i = 1;
        while (i < values.length) {
            int n = Math.min(blockSize, values.length - i);
            long[] d = new long[n];
            long min = Long.MAX_VALUE;
            for (int k = 0; k < n; k++) { d[k] = values[i + k] - values[i + k - 1]; min = Math.min(min, d[k]); }
            for (int k = 0; k < n; k++) d[k] -= min;
            int needed = (n + per - 1) / per;
            for (int m = 0; m < needed; m++) {
                long mask = 0;
                for (int k = m * per; k < Math.min(n, (m + 1) * per); k++) mask |= d[k];
                widths[m] = 64 - Long.numberOfLeadingZeros(mask);
            }
            writeZigZag(out, min);
            for (int m = 0; m < miniblocksPerBlock; m++) out.write(widths[m]);            // all widths, stale ones included
            for (int m = 0; m < needed; m++) {                                              // data only for miniblocks with values
                long[] mini = new long[per];
                for (int k = 0; k < per; k++) { int idx = m * per + k; mini[k] = idx < n ? d[idx] : 0; }
                writeBitPacked(out, widths[m], mini);
            }
            i += n;
        }
        return out.toByteArray();
    }

    static ByteReader reader(byte[] bytes) {
        ByteReader br = new ByteReader();
        br.wrap(bytes, 0, bytes.length);
        return br;
    }
}