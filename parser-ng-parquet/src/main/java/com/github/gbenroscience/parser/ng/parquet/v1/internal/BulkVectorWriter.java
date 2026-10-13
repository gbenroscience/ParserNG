package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import org.apache.arrow.memory.ArrowBuf;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

/**
 * Bulk, non-{@code setSafe}, non-per-element writes into an Arrow {@link FieldVector}'s validity and data
 * buffers, used by {@code ColumnPlan#writeSegment} instead of Arrow's standard {@code vector.set(index, value)}.
 *
 * <h2>Why this exists</h2>
 * {@code BaseFixedWidthVector#set(int, T)} -- the API this class replaces -- writes the data slot AND the
 * validity bit through {@link ArrowBuf}'s checked accessors ({@code setLong}/{@code setByte}/...), each of
 * which calls {@code ensureAccessible()} -> {@code refCnt()} (an atomic read) before every single element.
 * A CPU profile (not an allocation profile -- this is invisible to one) of a clean, fully-decoded,
 * SIMD-eligible scan showed this chain alone at roughly 50% of total CPU time, with the actual decode
 * kernel under 1%: the bottleneck was never decoding the page, it was writing the already-decoded values
 * out one safety-checked element at a time. This class does the same writes through {@link ArrowBuf}'s
 * bulk surface instead -- {@link ArrowBuf#nioBuffer(long, int)} for data (one bounds check for the whole
 * range, then a raw buffer-to-buffer transfer) and {@link ArrowBuf#setOne} plus a small number of
 * individually-checked boundary-byte reads for validity -- so the per-element check disappears without
 * giving up the validity bitmap's correctness.
 *
 * <h2>Validity bitmap: why boundary bytes still need a checked read</h2>
 * Row indices written across successive {@code fillBulk} calls are disjoint and monotonically increasing,
 * but a validity byte covers 8 rows, so the first and last byte touched by one call may be *shared* with a
 * neighboring call (e.g. rows [0,100) then [100,250): row 100's byte also holds rows 96-99, already
 * written by the first call). Blindly overwriting that byte would clobber those bits. Every byte fully
 * interior to the range has no such neighbor -- by construction nothing else will ever touch it -- so only
 * it is eligible for the unchecked bulk {@link ArrowBuf#setOne} path; the (at most two) boundary bytes are
 * still set bit-by-bit through the ordinary checked accessor, which is the same cost the old code paid for
 * every bit, just now bounded to O(1) calls per {@code writeSegment} invocation instead of O(n).
 *
 * <h2>Byte order</h2>
 * Every typed view below is forced to {@link ByteOrder#LITTLE_ENDIAN} explicitly rather than trusting the
 * platform's native order: Arrow buffers are little-endian by specification (and Parquet's PLAIN encoding,
 * which {@code SimdBulkDecode} already decodes assuming little-endian, matches it) regardless of what
 * {@code ByteBuffer.order()} would otherwise default to.
 */
final class BulkVectorWriter {
    private BulkVectorWriter() {}

    /**
     * Sets validity bits {@code [at, at+n)} to 1 (present). {@code buf} must already be zeroed beyond this
     * range (standard Arrow allocation behavior) so that unrelated bits -- including null rows this same
     * column will write later -- are left untouched by this call.
     */
    static void markValidRange(ArrowBuf buf, int at, int n) {
        if (n <= 0) return;
        int end = at + n; // exclusive
        int pos = at;

        // Head: finish out the first byte if 'at' isn't byte-aligned, bit by bit (checked; may be shared).
        int headBits = (8 - (pos & 7)) & 7;
        if (headBits > 0) {
            int take = Math.min(headBits, end - pos);
            for (int i = 0; i < take; i++) {
                org.apache.arrow.vector.BitVectorHelper.setBit(buf, pos + i);
            }
            pos += take;
        }

        // Middle: whole bytes entirely inside the range -- nothing else will ever touch them -- bulk-set.
        int wholeBytes = (end - pos) >>> 3;
        if (wholeBytes > 0) {
            buf.setOne((long) (pos >>> 3), wholeBytes);
            pos += wholeBytes * 8;
        }

        // Tail: remaining <8 bits of a byte that may be shared with a later call, bit by bit (checked).
        for (int i = pos; i < end; i++) {
            org.apache.arrow.vector.BitVectorHelper.setBit(buf, i);
        }
    }

    private static ByteBuffer rawView(ArrowBuf buf, long byteOffset, int byteLength) {
        return buf.nioBuffer(byteOffset, byteLength).order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Each {@code *View} factory below -- and every {@code put*} that takes a pre-made view -- exists for
     * one reason: a first version of this class called {@link ArrowBuf#nioBuffer} (and its {@code asXxxBuffer()}
     * follow-up) <em>inside</em> the per-run loop in {@code ColumnPlan.writeBulk}, i.e. once per contiguous
     * run of present values, not once per batch. That looked like it should still win -- call count dropped
     * roughly in proportion to average run length versus the one-call-per-element it replaced -- but a CPU
     * profile of the result showed {@code ByteBuffer.limit()} alone at 8% of total time, because creating
     * the {@code ByteBuffer}/{@code LongBuffer}/... wrapper object has its own real, allocation-bearing
     * cost, and at this engine's default null rates the average run is short enough (around 6 values at a
     * 15% null column) that paying that cost per run barely beats -- and on some scenarios lost to -- paying
     * Arrow's checked accessor once per element instead. The fix is not a different access mechanism, it's
     * calling {@code nioBuffer()} at a coarser grain: once per {@code writeSegment} invocation (i.e. once
     * per batch per column), covering the *whole* segment range, with every run inside that batch doing a
     * zero-allocation {@code put(index, array, offset, length)} -- the JDK's absolute bulk put, present on
     * every typed NIO buffer -- into the one already-built view. No cross-batch cache was needed to get
     * this grain: {@code RowGroupDecoder} calls {@code clear()}/{@code allocateNew()}/{@code reset()} on
     * every vector once per batch regardless, so the buffer this view wraps is never alive past the single
     * {@code writeSegment} call it was built for, and building a new one every such call is both correct
     * (nothing stale can ever be read through it) and, per this fix, now paid for once per batch instead of
     * once per run.
     */
    static LongBuffer longView(ArrowBuf data, int at, int len) {
        return rawView(data, (long) at * Long.BYTES, len * Long.BYTES).asLongBuffer();
    }

    static IntBuffer intView(ArrowBuf data, int at, int len) {
        return rawView(data, (long) at * Integer.BYTES, len * Integer.BYTES).asIntBuffer();
    }

    static FloatBuffer floatView(ArrowBuf data, int at, int len) {
        return rawView(data, (long) at * Float.BYTES, len * Float.BYTES).asFloatBuffer();
    }

    static DoubleBuffer doubleView(ArrowBuf data, int at, int len) {
        return rawView(data, (long) at * Double.BYTES, len * Double.BYTES).asDoubleBuffer();
    }

    static ShortBuffer shortView(ArrowBuf data, int at, int len) {
        return rawView(data, (long) at * Short.BYTES, len * Short.BYTES).asShortBuffer();
    }

    static ByteBuffer byteView(ArrowBuf data, int at, int len) {
        return rawView(data, at, len);
    }

    /** Narrowing int[] -> short write into an already-built view (INT16/UINT16; the decode kernel yields int[]). */
    static void putShortsNarrowing(ShortBuffer view, int viewOffset, int[] a, int base, int n) {
        for (int k = 0; k < n; k++) view.put(viewOffset + k, (short) a[base + k]);
    }

    /** Narrowing int[] -> byte write into an already-built view (INT8/UINT8; the decode kernel yields int[]). */
    static void putBytesNarrowing(ByteBuffer view, int viewOffset, int[] a, int base, int n) {
        for (int k = 0; k < n; k++) view.put(viewOffset + k, (byte) a[base + k]);
    }
}
