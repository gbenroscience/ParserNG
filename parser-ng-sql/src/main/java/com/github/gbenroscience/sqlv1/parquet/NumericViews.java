package com.github.gbenroscience.sqlv1.parquet;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Builds, per Parquet batch, the all-{@code Float8} view of the numeric columns
 * the expression engine needs.
 *
 * <p><b>Why.</b> {@code ArrowQuery} compiles double kernels only when every
 * column is a {@code Float8Vector}; Parquet batches carry native types (int32,
 * int64, float32, decimal, varchar, ...). Feeding them straight in selects the
 * float32 kernels and fails at binding. This class widens exactly the columns
 * an expression touches; everything else (strings, booleans, dates, timestamps,
 * nested) never enters the engine and is carried through natively by
 * {@link ParquetSqlPlan}.
 *
 * <p><b>Precision rule.</b> A double holds integers exactly up to 2^53. An
 * int64, uint64 or decimal unscaled value beyond that is never silently
 * rounded: {@link ParquetSqlOptions.Int64Policy#ERROR} (default) throws a
 * {@link ParquetSqlException} naming the column, row and value.
 *
 * <p>It also fills the <em>mask columns</em> of {@link NativeLeaves} (string, boolean, date and
 * timestamp predicates), so the engine sees only numeric leaves.
 *
 * <p><b>Nulls</b> are preserved exactly: a null slot is written with
 * {@code setNull}, never a sentinel value.
 *
 * <p><b>Allocation.</b> The widened vectors and the row-id vector are owned by
 * this object and reused across batches; they only grow. Float64 columns are
 * used by reference with no copy.
 */
final class NumericViews implements AutoCloseable {

    private static final long EXACT = 1L << 53;
    private static final double[] POW10 = new double[23];

    static {
        double p = 1;
        for (int i = 0; i < POW10.length; i++, p *= 10) POW10[i] = p;
    }

    /** Name of the synthetic row-index column used to gather passthrough columns. */
    static final String ROW_ID = "pngsqlrowid"; // letters only: see NativeLeaves.maskName

    private final BufferAllocator alloc;
    private final String[] names;
    private final ColumnKind[] kinds;
    private final ArrowType[] arrowTypes;
    private final Float8Vector[] owned;
    private final ParquetSqlOptions.Int64Policy policy;
    private final String file;
    private final boolean withRowId;
    private final List<NativeLeaves.Mask> masks;
    private final Float8Vector[] maskVecs;
    private Float8Vector rowId;
    private int rowIdFilled;
    private long roundedValues;

    NumericViews(BufferAllocator alloc, List<String> columns, TableSchema schema,
                 ParquetSqlOptions.Int64Policy policy, boolean withRowId, String file,
                 List<NativeLeaves.Mask> masks) {
        this.alloc = alloc;
        this.names = columns.toArray(new String[0]);
        this.kinds = new ColumnKind[names.length];
        this.arrowTypes = new ArrowType[names.length];
        this.owned = new Float8Vector[names.length];
        for (int i = 0; i < names.length; i++) {
            kinds[i] = schema.kind(names[i]);
            arrowTypes[i] = schema.type(names[i]);
            if (kinds[i] == null || !kinds[i].numeric()) {
                throw new ParquetSqlException("Column '" + names[i] + "' is " + kinds[i]
                        + " and cannot be used in a numeric expression");
            }
        }
        this.policy = policy;
        this.withRowId = withRowId;
        this.file = file;
        this.masks = masks;
        this.maskVecs = new Float8Vector[masks.size()];
    }

    /** @return number of values rounded under {@link ParquetSqlOptions.Int64Policy#ROUND_TO_DOUBLE} so far */
    long roundedValues() {
        return roundedValues;
    }

    /**
     * Widens the needed columns of {@code source} into an all-Float8 root.
     * The returned root is a holder only: do not close it; close this object.
     */
    VectorSchemaRoot bind(VectorSchemaRoot source) {
        int n = source.getRowCount();
        List<FieldVector> vecs = new ArrayList<>(names.length + 1);
        List<Field> fields = new ArrayList<>(names.length + 1);
        for (int c = 0; c < names.length; c++) {
            FieldVector src = source.getVector(names[c]);
            if (src == null) {
                throw new ParquetSqlException("Column '" + names[c] + "' is missing from the scan (file " + file + ")");
            }
            FieldVector v;
            if (kinds[c] == ColumnKind.FLOAT64) {
                v = src; // by reference, no copy
            } else {
                Float8Vector d = ensure(c, n);
                widen(c, src, d, n);
                d.setValueCount(n);
                v = d;
            }
            vecs.add(v);
            fields.add(v.getField());
        }
        VectorSchemaRoot widenedRoot = null; // holder over the widened columns, built only if a mask reads them
        for (int m = 0; m < maskVecs.length; m++) {
            Float8Vector mv = maskVecs[m];
            if (mv == null) {
                mv = new Float8Vector(masks.get(m).name, alloc);
                mv.allocateNew(Math.max(n, 1024));
                maskVecs[m] = mv;
            } else if (mv.getValueCapacity() < n) {
                mv.allocateNew(n);
            }
            NativeLeaves.Mask mk = masks.get(m);
            VectorSchemaRoot in = source;
            if (mk.widenedColumn() != null) {
                if (widenedRoot == null) {
                    widenedRoot = new VectorSchemaRoot(new Schema(new ArrayList<>(fields)), new ArrayList<>(vecs), n);
                }
                in = widenedRoot;
            }
            mk.eval(in, n, mv);
            mv.setValueCount(n);
            vecs.add(mv);
            fields.add(mv.getField());
        }
        if (withRowId) {
            if (rowId == null) {
                rowId = new Float8Vector(ROW_ID, alloc);
                rowId.allocateNew(Math.max(n, 1024));
                rowIdFilled = 0;
            } else if (rowId.getValueCapacity() < n) {
                rowId.allocateNew(n);
                rowIdFilled = 0;
            }
            for (int i = rowIdFilled; i < n; i++) rowId.set(i, i); // ids are 0..n-1: reuse the prefix
            rowIdFilled = Math.max(rowIdFilled, n);
            rowId.setValueCount(n);
            vecs.add(rowId);
            fields.add(rowId.getField());
        }
        return new VectorSchemaRoot(new Schema(fields), vecs, n);
    }

    private Float8Vector ensure(int c, int n) {
        Float8Vector d = owned[c];
        if (d == null) {
            d = new Float8Vector(names[c], alloc);
            d.allocateNew(Math.max(n, 1024));
            owned[c] = d;
        } else if (d.getValueCapacity() < n) {
            d.allocateNew(n);
        }
        return d;
    }

    private void widen(int c, FieldVector src, Float8Vector d, int n) {
        switch (kinds[c]) {
            case INT8 -> {
                TinyIntVector v = (TinyIntVector) src;
                for (int i = 0; i < n; i++) if (v.isNull(i)) d.setNull(i); else d.set(i, v.get(i));
            }
            case INT16 -> {
                SmallIntVector v = (SmallIntVector) src;
                for (int i = 0; i < n; i++) if (v.isNull(i)) d.setNull(i); else d.set(i, v.get(i));
            }
            case INT32 -> {
                IntVector v = (IntVector) src;
                for (int i = 0; i < n; i++) if (v.isNull(i)) d.setNull(i); else d.set(i, v.get(i));
            }
            case UINT8 -> {
                UInt1Vector v = (UInt1Vector) src;
                for (int i = 0; i < n; i++) if (v.isNull(i)) d.setNull(i); else d.set(i, v.get(i) & 0xFF);
            }
            case UINT16 -> {
                UInt2Vector v = (UInt2Vector) src;
                for (int i = 0; i < n; i++) if (v.isNull(i)) d.setNull(i); else d.set(i, v.get(i));
            }
            case UINT32 -> {
                UInt4Vector v = (UInt4Vector) src;
                for (int i = 0; i < n; i++) if (v.isNull(i)) d.setNull(i); else d.set(i, v.get(i) & 0xFFFFFFFFL);
            }
            case FLOAT32 -> {
                Float4Vector v = (Float4Vector) src;
                for (int i = 0; i < n; i++) if (v.isNull(i)) d.setNull(i); else d.set(i, v.get(i)); // float -> double is exact
            }
            case INT64 -> {
                BigIntVector v = (BigIntVector) src;
                for (int i = 0; i < n; i++) {
                    if (v.isNull(i)) { d.setNull(i); continue; }
                    long x = v.get(i);
                    if (x > EXACT || x < -EXACT) lossy(c, i, Long.toString(x));
                    d.set(i, (double) x);
                }
            }
            case UINT64 -> {
                UInt8Vector v = (UInt8Vector) src;
                for (int i = 0; i < n; i++) {
                    if (v.isNull(i)) { d.setNull(i); continue; }
                    long x = v.get(i);
                    if (x < 0 || x > EXACT) lossy(c, i, Long.toUnsignedString(x));
                    d.set(i, x >= 0 ? (double) x : (double) (x >>> 1) * 2.0);
                }
            }
            case DECIMAL -> widenDecimal(c, (DecimalVector) src, d, n);
            default -> throw new ParquetSqlException("Column '" + names[c] + "' is " + kinds[c]
                    + " and cannot be used in a numeric expression");
        }
    }

    private void widenDecimal(int c, DecimalVector v, Float8Vector d, int n) {
        ArrowType.Decimal t = (ArrowType.Decimal) arrowTypes[c];
        int scale = t.getScale();
        boolean fitsLong = t.getPrecision() <= 18;
        double div = scale >= 0 && scale < POW10.length ? POW10[scale] : Math.pow(10, scale);
        for (int i = 0; i < n; i++) {
            if (v.isNull(i)) { d.setNull(i); continue; }
            if (fitsLong) {
                long u = v.getDataBuffer().getLong((long) i * 16); // low 64 bits of the 128-bit LE value
                if (u > EXACT || u < -EXACT) lossy(c, i, Long.toString(u) + " (unscaled)");
                d.set(i, (double) u / div);
            } else {
                BigDecimal bd = v.getObject(i); // allocating slow path: precision > 18
                if (bd.unscaledValue().bitLength() > 53) lossy(c, i, bd.toPlainString());
                d.set(i, bd.doubleValue());
            }
        }
    }

    private void lossy(int c, int row, String value) {
        if (policy == ParquetSqlOptions.Int64Policy.ROUND_TO_DOUBLE) {
            roundedValues++;
            return;
        }
        throw new ParquetSqlException("Column '" + names[c] + "' (" + kinds[c] + ") has value " + value
                + " at batch row " + row + " (file " + file + ") beyond 2^53, which a double cannot hold exactly. "
                + "Refusing to round silently. Select the column without arithmetic, restrict the rows, "
                + "or opt in with ParquetSqlOptions.withInt64Policy(ROUND_TO_DOUBLE).");
    }

    @Override
    public void close() {
        for (Float8Vector v : owned) if (v != null) v.close();
        for (Float8Vector v : maskVecs) if (v != null) v.close();
        if (rowId != null) rowId.close();
    }
}
