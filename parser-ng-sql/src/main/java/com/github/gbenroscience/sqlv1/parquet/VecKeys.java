package com.github.gbenroscience.sqlv1.parquet;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;

/** Allocation-free access to the values the Java aggregation paths hash and compare. */
final class VecKeys {

    private VecKeys() { }

    /** True if {@link #longOf} can read {@code v}: integers, dates, timestamps, booleans and floating point. */
    static boolean hasLongKey(FieldVector v) {
        return v instanceof IntVector || v instanceof BigIntVector || v instanceof SmallIntVector || v instanceof TinyIntVector
                || v instanceof UInt1Vector || v instanceof UInt2Vector || v instanceof UInt4Vector || v instanceof UInt8Vector
                || v instanceof DateDayVector || v instanceof DateMilliVector || v instanceof TimeStampVector
                || v instanceof BitVector || v instanceof Float8Vector || v instanceof Float4Vector;
    }

    /** The value of a non-null row as a {@code long}; floating point as raw bits with {@code -0.0} folded into {@code 0.0}. */
    static long longOf(FieldVector v, int row) {
        if (v instanceof IntVector c) return c.get(row);
        if (v instanceof BigIntVector c) return c.get(row);
        if (v instanceof SmallIntVector c) return c.get(row);
        if (v instanceof TinyIntVector c) return c.get(row);
        if (v instanceof UInt1Vector c) return c.get(row) & 0xFFL;
        if (v instanceof UInt2Vector c) return c.get(row) & 0xFFFFL;
        if (v instanceof UInt4Vector c) return c.get(row) & 0xFFFFFFFFL;
        if (v instanceof UInt8Vector c) return c.get(row);
        if (v instanceof DateDayVector c) return c.get(row);
        if (v instanceof DateMilliVector c) return c.get(row);
        if (v instanceof TimeStampVector c) return c.get(row);
        if (v instanceof BitVector c) return c.get(row);
        if (v instanceof Float8Vector c) return LongKeyTable.ofDouble(c.get(row));
        if (v instanceof Float4Vector c) return LongKeyTable.ofDouble(c.get(row));
        throw new ParquetSqlException("Vector type " + v.getClass().getSimpleName() + " has no integer key");
    }

    /** Compares the bytes of {@code row} with {@code best} in UTF-8 (unsigned byte) order, without copying the row. */
    static int compareRow(VarCharVector s, int row, byte[] best) {
        int start = s.getStartOffset(row);
        int len = s.getEndOffset(row) - start;
        ArrowBuf data = s.getDataBuffer();
        int n = Math.min(len, best.length);
        for (int k = 0; k < n; k++) {
            int d = (data.getByte(start + k) & 0xFF) - (best[k] & 0xFF);
            if (d != 0) return d;
        }
        return Integer.compare(len, best.length);
    }
}
