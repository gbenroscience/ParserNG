package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo.ColumnChunkInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo.RowGroupInfo;
import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code COUNT(col)}, {@code MIN(col)} and {@code MAX(col)} (with {@code COUNT(*)}) answered from the Parquet
 * footer, reading no data page, when the answer is <em>provably</em> what a scan would compute.
 *
 * <p><b>When it applies.</b> No {@code WHERE}, {@code GROUP BY}, {@code HAVING}, {@code ORDER BY} or
 * {@code DISTINCT}; every select item is {@code COUNT(*)}, {@code COUNT(col)}, {@code MIN(col)} or
 * {@code MAX(col)} over a plain top-level flat column. Anything else, and any doubt, falls back to the
 * normal path: this class only ever returns an answer or {@code null}.
 *
 * <p><b>When it is exact.</b>
 * <ul>
 * <li>{@code COUNT(col)} needs a null count for every row group of every file; the answer is values minus nulls.</li>
 * <li>{@code MIN}/{@code MAX} are taken from the footer only for signed integers, {@code DATE} and
 * {@code BOOLEAN}, whose statistics are exact by definition. {@code FLOAT} and {@code DOUBLE} are always
 * scanned: a writer may record {@code -0.0} as the minimum of a column whose smallest value is {@code 0.0}
 * and leaves {@code NaN} out of its statistics, so the footer can disagree with a scan. Strings and binary
 * (statistics may be truncated or written with a wrong order by old writers), decimals, unsigned integers
 * and timestamps (statistics are rendered in a writer-dependent text form) are always scanned.</li>
 * <li>Every row group must either carry usable min and max, or be all nulls (null count equal to the row
 * count). A row group with neither makes the whole answer unknown.</li>
 * <li>A 64-bit integer beyond 2^53 and a {@code NaN} statistic fall back, so the scan decides.</li>
 * </ul>
 *
 * <p>Results use the engine's types: {@code Float8} for counts and numeric {@code MIN}/{@code MAX}, the
 * column's own type for {@code DATE} and {@code BOOLEAN}. A column with no values gives {@code COUNT} 0 and a
 * null {@code MIN}/{@code MAX}. The footer is read once, when the plan is built.
 */
final class FooterAggregates {

    private static final double TWO_POW_53 = 9007199254740992.0;

    private FooterAggregates() { }

    private enum How { COUNT_STAR, COUNT_COL, MINMAX }

    /** The answer: one value per select item. */
    static final class Answer {
        private final List<String> names;
        private final List<How> hows;
        private final List<ColumnKind> kinds;
        private final List<Double> numbers;   // COUNT and numeric MIN/MAX
        private final List<Long> longs;       // DATE (epoch day) and BOOLEAN (0/1)

        private Answer(List<String> names, List<How> hows, List<ColumnKind> kinds, List<Double> numbers, List<Long> longs) {
            this.names = names;
            this.hows = hows;
            this.kinds = kinds;
            this.numbers = numbers;
            this.longs = longs;
        }

        VectorSchemaRoot toRoot(BufferAllocator a) {
            List<FieldVector> vecs = new ArrayList<>();
            try {
                for (int i = 0; i < names.size(); i++) {
                    ColumnKind k = kinds.get(i);
                    boolean asLong = hows.get(i) == How.MINMAX && (k == ColumnKind.DATE || k == ColumnKind.BOOL);
                    if (!asLong) {
                        Float8Vector v = new Float8Vector(names.get(i), a);
                        v.allocateNew(1);
                        if (numbers.get(i) != null) v.set(0, numbers.get(i));
                        v.setValueCount(1);
                        vecs.add(v);
                    } else if (k == ColumnKind.DATE) {
                        DateDayVector v = new DateDayVector(names.get(i), a);
                        v.allocateNew(1);
                        if (longs.get(i) != null) v.set(0, (int) (long) longs.get(i));
                        v.setValueCount(1);
                        vecs.add(v);
                    } else {
                        BitVector v = new BitVector(names.get(i), a);
                        v.allocateNew(1);
                        if (longs.get(i) != null) v.set(0, (int) (long) longs.get(i));
                        v.setValueCount(1);
                        vecs.add(v);
                    }
                }
            } catch (RuntimeException e) {
                for (FieldVector v : vecs) v.close();
                throw e;
            }
            VectorSchemaRoot r = new VectorSchemaRoot(vecs);
            r.setRowCount(1);
            return r;
        }
    }

    /** @return the footer answer, or {@code null} if the query is not of this shape or the footer cannot prove it */
    static Answer tryAnswer(SelectStatement st, TableSchema schema, List<ParquetFileInfo> infos) {
        if (st.selectAll() || st.where() != null || !st.groupBy().isEmpty() || st.having() != null
                || !st.orderBy().isEmpty() || st.distinct() || infos.isEmpty()) {
            return null;
        }
        List<String> names = new ArrayList<>();
        List<How> hows = new ArrayList<>();
        List<ColumnKind> kinds = new ArrayList<>();
        List<Double> numbers = new ArrayList<>();
        List<Long> longs = new ArrayList<>();
        long totalRows = 0;
        for (ParquetFileInfo info : infos) totalRows += info.rowCount();

        for (SelectItem it : st.items()) {
            if (!it.isAggregate()) return null;
            AggregateSpec a = it.aggregate();
            if (a.distinct()) return null;
            names.add(it.outputName());
            if (a.star()) {
                if (a.func() != AggFunc.COUNT) return null;
                hows.add(How.COUNT_STAR);
                kinds.add(null);
                numbers.add((double) totalRows);
                longs.add(null);
                continue;
            }
            if (a.func() != AggFunc.COUNT && a.func() != AggFunc.MIN && a.func() != AggFunc.MAX) return null;
            String ref = a.argExprText().trim();
            if (!schema.hasFileColumn(ref)) return null;
            String col = QuotedIdentifiers.real(ref);
            ColumnKind k = schema.kind(ref);
            for (ParquetFileInfo info : infos) if (!info.primitiveColumns().contains(col)) return null;

            if (a.func() == AggFunc.COUNT) {
                Long nonNull = nonNullCount(infos, col);
                if (nonNull == null) return null;
                hows.add(How.COUNT_COL);
                kinds.add(k);
                numbers.add((double) nonNull);
                longs.add(null);
                continue;
            }
            if (!supportedExtreme(k)) return null;
            Extreme e = extreme(infos, col, k, a.func() == AggFunc.MAX);
            if (e == null) return null;
            hows.add(How.MINMAX);
            kinds.add(k);
            if (k == ColumnKind.DATE || k == ColumnKind.BOOL) {
                numbers.add(null);
                longs.add(e.present ? e.asLong : null);
            } else {
                numbers.add(e.present ? e.asDouble : null);
                longs.add(null);
            }
        }
        return new Answer(names, hows, kinds, numbers, longs);
    }

    private static boolean supportedExtreme(ColumnKind k) {
        return switch (k) {
            case INT8, INT16, INT32, INT64, DATE, BOOL -> true;
            default -> false;
        };
    }

    private static ColumnChunkInfo chunk(RowGroupInfo rg, String col) {
        for (ColumnChunkInfo c : rg.columns()) if (c.path().equals(col)) return c;
        return null;
    }

    /** Values minus nulls over every row group, or {@code null} if any row group has no null count. */
    private static Long nonNullCount(List<ParquetFileInfo> infos, String col) {
        long n = 0;
        for (ParquetFileInfo info : infos) {
            for (RowGroupInfo rg : info.rowGroups()) {
                ColumnChunkInfo c = chunk(rg, col);
                if (c == null || c.nullCount() == null) return null;
                n += rg.rowCount() - c.nullCount();
            }
        }
        return n;
    }

    private static final class Extreme {
        boolean present;
        double asDouble;
        long asLong;
    }

    /** The overall min or max of {@code col}, or {@code null} if the footer cannot prove it. */
    private static Extreme extreme(List<ParquetFileInfo> infos, String col, ColumnKind k, boolean wantMax) {
        Extreme out = new Extreme();
        for (ParquetFileInfo info : infos) {
            for (RowGroupInfo rg : info.rowGroups()) {
                if (rg.rowCount() == 0) continue;
                ColumnChunkInfo c = chunk(rg, col);
                if (c == null) return null;
                String text = wantMax ? c.max() : c.min();
                if (text == null) {
                    // no usable statistics: only fine if every value in the chunk is null
                    if (c.nullCount() != null && c.nullCount() == rg.rowCount()) continue;
                    return null;
                }
                double d;
                long l;
                try {
                    if (k == ColumnKind.DATE) {
                        l = text.indexOf('-', 1) > 0 ? LocalDate.parse(text).toEpochDay() : Long.parseLong(text);
                        d = l;
                    } else if (k == ColumnKind.BOOL) {
                        if (!text.equalsIgnoreCase("true") && !text.equalsIgnoreCase("false")) return null;
                        l = text.equalsIgnoreCase("true") ? 1 : 0;
                        d = l;
                    } else {
                        l = Long.parseLong(text);
                        d = l;
                        if (k == ColumnKind.INT64 && Math.abs((double) l) > TWO_POW_53) return null; // the scan's policy decides
                    }
                } catch (RuntimeException unparsable) {
                    return null;
                }
                if (Double.isNaN(d)) return null;
                if (!out.present || (wantMax ? d > out.asDouble : d < out.asDouble)) {
                    out.present = true;
                    out.asDouble = d;
                    out.asLong = l;
                }
            }
        }
        return out;
    }
}
