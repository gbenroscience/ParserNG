package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import org.apache.parquet.bytes.BytesInput;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * A view of a page payload as {@code (array, offset, length)}. Heap-backed, writable payloads are exposed
 * in place (no copy); anything else (direct, read-only, or multi-slice) is copied once, exactly as
 * {@code BytesInput#toByteArray()} would have. The view is valid until the owning row group is released,
 * which outlives every use in {@link FastColumnCursor} (one cursor per column per row group).
 */
final class PageBytes {
    final byte[] buf;
    final int off;
    final int len;

    private PageBytes(byte[] buf, int off, int len) {
        this.buf = buf;
        this.off = off;
        this.len = len;
    }

    static PageBytes of(BytesInput in) throws IOException {
        ByteBuffer bb = in.toByteBuffer();
        if (bb.hasArray()) {
            return new PageBytes(bb.array(), bb.arrayOffset() + bb.position(), bb.remaining());
        }
        byte[] copy = new byte[bb.remaining()];
        bb.duplicate().get(copy);
        return new PageBytes(copy, 0, copy.length);
    }
}
