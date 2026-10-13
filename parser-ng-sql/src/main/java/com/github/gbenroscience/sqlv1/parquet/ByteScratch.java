package com.github.gbenroscience.sqlv1.parquet;

import org.apache.arrow.vector.VarCharVector;

/** A reusable byte buffer that a string value is copied into, so it can be hashed or compared without allocating. */
final class ByteScratch {

    byte[] buf = new byte[64];

    /** Copies the bytes of {@code row} (not null) into {@link #buf} and returns their count. */
    int load(VarCharVector v, int row) {
        int start = v.getStartOffset(row);
        int len = v.getEndOffset(row) - start;
        if (len > buf.length) buf = new byte[Math.max(len, buf.length * 2)];
        if (len > 0) v.getDataBuffer().getBytes(start, buf, 0, len);
        return len;
    }
}
