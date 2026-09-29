package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import java.util.Arrays;

/**
 * Decodes DELTA_BINARY_PACKED (Parquet encoding id 5): the frame-of-reference delta encoding used for
 * INT32/INT64 columns, and as the length/prefix/suffix-length sub-stream inside DELTA_LENGTH_BYTE_ARRAY
 * and DELTA_BYTE_ARRAY (see {@link DeltaByteArrayDecoder}). Format, verified against the parquet-format
 * spec (Encodings.md, "Delta Encoding (DELTA_BINARY_PACKED = 5)"):
 *
 * <pre>
 * header := &lt;block size in values&gt; &lt;miniblocks per block&gt; &lt;total value count&gt; &lt;first value&gt;
 * block   := &lt;min delta&gt; &lt;bitwidth of each miniblock, one byte each&gt; &lt;miniblocks&gt;
 * </pre>
 *
 * block size / miniblock count / value count are ULEB128; the first value and each block's min delta
 * are zigzag ULEB128; each miniblock is {@code blockSizeInValues / miniblocksPerBlock} values, LSB-first
 * bit-packed (same packing convention as an RLE/bit-packing-hybrid bit-packed run) at that miniblock's
 * own bit width. A final, partially-filled block still carries a full header and bitwidth byte for
 * every miniblock -- the spec's "when there are not enough values to encode a full block we pad with
 * zeros" -- so this decoder always reads {@code miniblocksPerBlock} miniblocks per block and lets the
 * padding miniblocks' bit width (0, in every writer this was checked against) account for their absence
 * of data bytes, rather than trying to infer how many miniblocks were "really" written.
 *
 * <p>Only the page's <em>present</em> entries are delta-encoded (nulls carry no value, exactly as for
 * every other value stream {@link FastColumnCursor} reads), so {@code count} here is always the page's
 * present-entry count, never its total entry count.
 *
 * <p>Correctness-first, not benchmarked: allocates two small scratch arrays per call (bit widths and
 * one miniblock's worth of deltas) rather than threading caller-owned scratch through, unlike the rest
 * of this decode engine's scratch-reuse convention. Revisit if profiling ever shows this on a hot path.
 */
public final class DeltaBinaryPackedDecoder {

    private DeltaBinaryPackedDecoder() { }

    /**
     * Decodes exactly {@code count} values into {@code out[0, count)}, advancing {@code br} past the
     * entire delta-encoded section -- header, every block, every miniblock's padding included -- so a
     * caller with more data following in the same page (see {@link DeltaByteArrayDecoder}) can keep
     * reading from {@code br.pos} afterward.
     */
    public static void decode(ByteReader br, long[] out, int count) {
        final int blockSizeInValues = br.readUnsignedVarInt();
        final int miniblocksPerBlock = br.readUnsignedVarInt();
        final int totalValueCount = br.readUnsignedVarInt();
        final long firstValue = ByteReader.zigZagDecode(br.readUnsignedVarLong());

        if (totalValueCount != count) {
            throw new ByteReader.Corrupt(br.pos, 0, br.limit, br.buf.length);
        }
        if (count == 0) return;
        out[0] = firstValue;
        if (count == 1) return;
        if (miniblocksPerBlock <= 0 || blockSizeInValues <= 0 || blockSizeInValues % miniblocksPerBlock != 0) {
            throw new ByteReader.Corrupt(br.pos, 0, br.limit, br.buf.length);
        }

        final int valuesPerMiniblock = blockSizeInValues / miniblocksPerBlock;
        if (valuesPerMiniblock <= 0 || valuesPerMiniblock % 8 != 0) {
            throw new ByteReader.Corrupt(br.pos, 0, br.limit, br.buf.length);
        }
        final int groupCount8 = valuesPerMiniblock / 8;

        final int[] bitWidths = new int[miniblocksPerBlock];
        final long[] deltaScratch = new long[valuesPerMiniblock];

        long current = firstValue;
        int filled = 1;
        int remaining = count - 1;

        while (remaining > 0) {
            final long minDelta = ByteReader.zigZagDecode(br.readUnsignedVarLong());
            for (int m = 0; m < miniblocksPerBlock; m++) {
                final int w = br.readByte() & 0xFF;
                if (w > 64) throw new ByteReader.Corrupt(br.pos, 0, br.limit, br.buf.length);
                bitWidths[m] = w;
            }
            // Every miniblock in the block is present in the stream (see class Javadoc), so this loop
            // never breaks early on `remaining` -- an already-exhausted miniblock still costs a
            // readBitPackedGroupsLong call, but at bit width 0 (the padding convention) that call reads
            // zero bytes and writes only zeros, so it is a correct no-op rather than a special case.
            for (int m = 0; m < miniblocksPerBlock; m++) {
                br.readBitPackedGroupsLong(bitWidths[m], groupCount8, deltaScratch, 0, valuesPerMiniblock);
                final int take = Math.max(0, Math.min(remaining, valuesPerMiniblock));
                for (int i = 0; i < take; i++) {
                    current += minDelta + deltaScratch[i];
                    out[filled++] = current;
                }
                remaining -= take;
            }
        }
    }

    /** Convenience for the (very common) INT32 case: decodes into a {@code long[]} scratch, then narrows. */
    public static void decodeInts(ByteReader br, long[] longScratch, int[] out, int count) {
        decode(br, longScratch, count);
        for (int i = 0; i < count; i++) out[i] = (int) longScratch[i];
    }
}
