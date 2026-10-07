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
 * own bit width. A final, partially-filled block still carries a bitwidth byte for <em>every</em>
 * miniblock, but the <em>data</em> of miniblocks that hold no values is omitted. The bit widths of those
 * unneeded miniblocks are <b>not reliably zero</b>: parquet-java's writer keeps its width array across
 * blocks and only refreshes the entries it needs, so after a full block the unused slots still hold the
 * previous block's widths. This decoder therefore reads data for exactly
 * {@code ceil(remainingValues / valuesPerMiniblock)} miniblocks (capped at {@code miniblocksPerBlock}),
 * which is what parquet-java's own reader does, and never trusts the width of a miniblock it does not need.
 * (An earlier version read every miniblock and relied on padding widths being 0; it mis-parsed any page whose
 * delta stream had a full block followed by a short one, which showed up as a bogus value-count mismatch
 * on the <em>second</em> delta sequence of a DELTA_BYTE_ARRAY page.)
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
            // Only miniblocks that hold values have data in the stream; the widths of the rest are present but
            // meaningless (see class Javadoc), so they are neither read nor trusted.
            final int needed = (int) Math.min((long) miniblocksPerBlock, ((long) remaining + valuesPerMiniblock - 1) / valuesPerMiniblock);
            for (int m = 0; m < needed; m++) {
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