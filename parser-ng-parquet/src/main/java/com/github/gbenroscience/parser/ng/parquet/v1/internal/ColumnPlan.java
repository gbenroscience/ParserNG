package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScanException;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.FastColumnCursor;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.LogicalTypeAnnotation.*;
import org.apache.parquet.schema.PrimitiveType;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.file.Path;

/**
 * Per-column mapping Parquet -> Arrow plus the primitive decode loop.
 *
 * <p>Scope of this class (flat, non-repeated primitive columns only): anything else throws a
 * {@link ParquetScanException} naming the column and the reason. Nothing is ever coerced silently.
 *
 * <p>Hot loop: no boxing. A null is left as an unset validity bit; the value slot is never written,
 * so null is never turned into 0/false/"".
 *
 * <h2>Two decode loops</h2>
 * <ul>
 *   <li><b>Bulk</b> (fixed-width numeric kinds when the reader is a {@link FastColumnCursor}): the cursor hands
 *       over one page segment at a time as decoded arrays plus definition levels, and each segment is written
 *       to Arrow in a single tight typed loop -- no per-value {@code consume()}, getter or physical-type
 *       {@code switch}. Nullable columns scatter the dense present values by definition level.</li>
 *   <li><b>Per value</b> (BOOL, UTF8, BINARY, or any other {@link ColumnReader} implementation): one virtual
 *       {@code ColumnReader} call per value plus one direct Arrow {@code set}.</li>
 * </ul>
 * This class is stateless and shared across decoder threads; all scratch lives in the cursor.
 */
public final class ColumnPlan {

    enum Kind { BOOL, INT8, INT16, INT32, DATE_DAY, INT64, TIMESTAMP, FLOAT, DOUBLE, UTF8, BINARY, DECIMAL,
                UINT8, UINT16, UINT32, UINT64, UUID }

    /** Which physical type a DECIMAL-kind column's unscaled value is read from; meaningless for every other {@link Kind}. */
    private enum DecimalSource { INT32, INT64, BYTES }

    private final Kind kind;
    private final int maxDef;
    private final Field field;
    private final DecimalSource decimalSource;
    private final int decimalScale;

    private ColumnPlan(Kind kind, int maxDef, Field field) {
        this(kind, maxDef, field, null, 0);
    }

    private ColumnPlan(Kind kind, int maxDef, Field field, DecimalSource decimalSource, int decimalScale) {
        this.kind = kind;
        this.maxDef = maxDef;
        this.field = field;
        this.decimalSource = decimalSource;
        this.decimalScale = decimalScale;
    }

    public Field field() { return field; }

    /** Flat (top-level, non-repeated) column. Anything nested is planned by {@link NodePlan} instead. */
    public static ColumnPlan of(ColumnDescriptor d, Path file) {
        String name = d.getPath()[0];
        if (d.getPath().length > 1) throw unsupported(file, name, "nested column reached the flat planner");
        if (d.getMaxRepetitionLevel() > 0) throw unsupported(file, name, "repeated column reached the flat planner");
        return forPrimitive(d.getPrimitiveType(), name, d.getMaxDefinitionLevel() > 0, d.getMaxDefinitionLevel(), file);
    }

    /**
     * Type mapping for any primitive leaf, flat or nested.
     *
     * @param maxDef the leaf's maximum definition level (a value is present iff def >= maxDef)
     */
    public static ColumnPlan forPrimitive(PrimitiveType pt, String name, boolean nullable, int maxDef, Path file) {
        LogicalTypeAnnotation lt = pt.getLogicalTypeAnnotation();
        Kind k;
        ArrowType at;
        DecimalSource decSrc = null;
        int decScale = 0;
        switch (pt.getPrimitiveTypeName()) {
            case BOOLEAN:
                needNoLogical(lt, file, name);
                k = Kind.BOOL; at = ArrowType.Bool.INSTANCE; break;
            case INT32:
                if (lt == null) { k = Kind.INT32; at = new ArrowType.Int(32, true); }
                else if (lt instanceof DateLogicalTypeAnnotation) { k = Kind.DATE_DAY; at = new ArrowType.Date(DateUnit.DAY); }
                else if (lt instanceof IntLogicalTypeAnnotation i && i.isSigned() && i.getBitWidth() == 32) { k = Kind.INT32; at = new ArrowType.Int(32, true); }
                else if (lt instanceof IntLogicalTypeAnnotation i && i.isSigned() && i.getBitWidth() == 16) { k = Kind.INT16; at = new ArrowType.Int(16, true); }
                else if (lt instanceof IntLogicalTypeAnnotation i && i.isSigned() && i.getBitWidth() == 8) { k = Kind.INT8; at = new ArrowType.Int(8, true); }
                else if (lt instanceof DecimalLogicalTypeAnnotation dec) {
                    k = Kind.DECIMAL; at = new ArrowType.Decimal(dec.getPrecision(), dec.getScale(), 128);
                    decSrc = DecimalSource.INT32; decScale = dec.getScale();
                }
                // Unsigned INT(8|16|32, false) are stored as INT32 physical (parquet-format
                // LogicalTypes.md, "Unsigned Integers"); UINT_64 is INT64 physical -- see that case below.
                // Decoded exactly like their signed counterparts (PLAIN/dictionary/DELTA_BINARY_PACKED
                // are bit-pattern-agnostic to signedness); only the resulting Arrow vector type differs.
                else if (lt instanceof IntLogicalTypeAnnotation i && !i.isSigned() && i.getBitWidth() == 32) { k = Kind.UINT32; at = new ArrowType.Int(32, false); }
                else if (lt instanceof IntLogicalTypeAnnotation i && !i.isSigned() && i.getBitWidth() == 16) { k = Kind.UINT16; at = new ArrowType.Int(16, false); }
                else if (lt instanceof IntLogicalTypeAnnotation i && !i.isSigned() && i.getBitWidth() == 8) { k = Kind.UINT8; at = new ArrowType.Int(8, false); }
                else throw unsupported(file, name, "INT32 with logical type " + lt + " (TIME not supported yet)");
                break;
            case INT64:
                if (lt == null || (lt instanceof IntLogicalTypeAnnotation i && i.isSigned() && i.getBitWidth() == 64)) {
                    k = Kind.INT64; at = new ArrowType.Int(64, true);
                } else if (lt instanceof TimestampLogicalTypeAnnotation ts) {
                    k = Kind.TIMESTAMP;
                    TimeUnit u;
                    switch (ts.getUnit()) {
                        case MILLIS: u = TimeUnit.MILLISECOND; break;
                        case MICROS: u = TimeUnit.MICROSECOND; break;
                        default: u = TimeUnit.NANOSECOND; break;
                    }
                    at = new ArrowType.Timestamp(u, ts.isAdjustedToUTC() ? "UTC" : null);
                } else if (lt instanceof DecimalLogicalTypeAnnotation dec) {
                    k = Kind.DECIMAL; at = new ArrowType.Decimal(dec.getPrecision(), dec.getScale(), 128);
                    decSrc = DecimalSource.INT64; decScale = dec.getScale();
                } else if (lt instanceof IntLogicalTypeAnnotation i && !i.isSigned() && i.getBitWidth() == 64) {
                    k = Kind.UINT64; at = new ArrowType.Int(64, false);
                } else throw unsupported(file, name, "INT64 with logical type " + lt + " (TIME not supported yet)");
                break;
            case INT96:
                // Deprecated legacy timestamp type; no LogicalTypeAnnotation exists for it (it predates
                // that mechanism). See FastColumnCursor's INT96 case / Int96Timestamp for the byte layout
                // this assumes and the nanos-since-epoch conversion every reader here relies on.
                needNoLogical(lt, file, name);
                k = Kind.TIMESTAMP; at = new ArrowType.Timestamp(TimeUnit.NANOSECOND, "UTC"); break;
            case FLOAT:
                needNoLogical(lt, file, name);
                k = Kind.FLOAT; at = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE); break;
            case DOUBLE:
                needNoLogical(lt, file, name);
                k = Kind.DOUBLE; at = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE); break;
            case BINARY:
                if (lt == null) { k = Kind.BINARY; at = ArrowType.Binary.INSTANCE; }
                else if (lt instanceof StringLogicalTypeAnnotation || lt instanceof EnumLogicalTypeAnnotation
                        || lt instanceof JsonLogicalTypeAnnotation) { k = Kind.UTF8; at = ArrowType.Utf8.INSTANCE; }
                else if (lt instanceof DecimalLogicalTypeAnnotation dec) {
                    k = Kind.DECIMAL; at = new ArrowType.Decimal(dec.getPrecision(), dec.getScale(), 128);
                    decSrc = DecimalSource.BYTES; decScale = dec.getScale();
                }
                else throw unsupported(file, name, "BINARY with logical type " + lt + " (BSON not supported yet)");
                break;
            case FIXED_LEN_BYTE_ARRAY:
                if (lt instanceof DecimalLogicalTypeAnnotation dec) {
                    k = Kind.DECIMAL; at = new ArrowType.Decimal(dec.getPrecision(), dec.getScale(), 128);
                    decSrc = DecimalSource.BYTES; decScale = dec.getScale();
                } else if (lt instanceof UUIDLogicalTypeAnnotation) {
                    // UUID annotates FIXED_LEN_BYTE_ARRAY(16) only (parquet-format LogicalTypes.md); a
                    // writer claiming UUID on any other width is itself non-conformant, so this is a
                    // corrupt/unsupported-file error, not a "not implemented yet" one.
                    if (pt.getTypeLength() != 16) {
                        throw unsupported(file, name, "UUID logical type on a FIXED_LEN_BYTE_ARRAY(" + pt.getTypeLength() + ") column (must be 16)");
                    }
                    k = Kind.UUID; at = new ArrowType.FixedSizeBinary(16);
                } else throw unsupported(file, name, "FIXED_LEN_BYTE_ARRAY with logical type " + lt + " (only DECIMAL/UUID supported)");
                break;
            default:
                throw unsupported(file, name, pt.getPrimitiveTypeName() + " (not supported yet)");
        }
        return new ColumnPlan(k, maxDef, new Field(name, new FieldType(nullable, at, null), null), decSrc, decScale);
    }

    private static void needNoLogical(LogicalTypeAnnotation lt, Path file, String name) {
        if (lt != null) throw unsupported(file, name, "logical type " + lt + " on this physical type");
    }

    private static ParquetScanException unsupported(Path file, String col, String why) {
        return new ParquetScanException("Unsupported Parquet column: " + why, file, -1, col, null);
    }

    /** Fills {@code v[0..n)} from {@code cr}; the caller sets value count / row count. */
    public void fill(ColumnReader cr, int n, FieldVector v) {
        if (cr instanceof FastColumnCursor fc && fc.supportsBulk() && isBulkKind()) {
            fillBulk(fc, n, v);
            return;
        }
        final int maxDef = this.maxDef;
        switch (kind) {
            case INT32: {
                IntVector o = (IntVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getInteger());
                    cr.consume();
                }
                break;
            }
            case INT64: {
                BigIntVector o = (BigIntVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getLong());
                    cr.consume();
                }
                break;
            }
            case DOUBLE: {
                Float8Vector o = (Float8Vector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getDouble());
                    cr.consume();
                }
                break;
            }
            case FLOAT: {
                Float4Vector o = (Float4Vector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getFloat());
                    cr.consume();
                }
                break;
            }
            case BOOL: {
                BitVector o = (BitVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getBoolean() ? 1 : 0);
                    cr.consume();
                }
                break;
            }
            case INT8: {
                TinyIntVector o = (TinyIntVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, (byte) cr.getInteger());
                    cr.consume();
                }
                break;
            }
            case INT16: {
                SmallIntVector o = (SmallIntVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, (short) cr.getInteger());
                    cr.consume();
                }
                break;
            }
            case DATE_DAY: {
                DateDayVector o = (DateDayVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getInteger());
                    cr.consume();
                }
                break;
            }
            case TIMESTAMP: {
                TimeStampVector o = (TimeStampVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getLong());
                    cr.consume();
                }
                break;
            }
            case UTF8: {
                VarCharVector o = (VarCharVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) {
                        ByteBuffer bb = cr.getBinary().toByteBuffer();
                        o.setSafe(i, bb, bb.position(), bb.remaining());
                    }
                    cr.consume();
                }
                break;
            }
            case BINARY: {
                VarBinaryVector o = (VarBinaryVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) {
                        ByteBuffer bb = cr.getBinary().toByteBuffer();
                        o.setSafe(i, bb, bb.position(), bb.remaining());
                    }
                    cr.consume();
                }
                break;
            }
            case DECIMAL: {
                DecimalVector o = (DecimalVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, decimalOf(cr));
                    cr.consume();
                }
                break;
            }
            // Unsigned ints: identical decode to their signed counterparts (see forPrimitive's
            // Javadoc note) -- only the Arrow vector type differs, so these mirror INT8/INT16/INT32/
            // INT64 exactly, just writing into Uint1Vector/Uint2Vector/Uint4Vector/Uint8Vector.
            case UINT8: {
                UInt1Vector o = (UInt1Vector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, (byte) cr.getInteger());
                    cr.consume();
                }
                break;
            }
            case UINT16: {
                UInt2Vector o = (UInt2Vector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, (short) cr.getInteger());
                    cr.consume();
                }
                break;
            }
            case UINT32: {
                UInt4Vector o = (UInt4Vector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getInteger());
                    cr.consume();
                }
                break;
            }
            case UINT64: {
                UInt8Vector o = (UInt8Vector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, cr.getLong());
                    cr.consume();
                }
                break;
            }
            case UUID: {
                FixedSizeBinaryVector o = (FixedSizeBinaryVector) v;
                for (int i = 0; i < n; i++) {
                    if (maxDef == 0 || cr.getCurrentDefinitionLevel() == maxDef) o.set(i, uuidBytes(cr));
                    cr.consume();
                }
                break;
            }
            default:
                throw new IllegalStateException("unhandled kind " + kind);
        }
    }

    /** Copies the current FIXED_LEN_BYTE_ARRAY(16) value's raw bytes out; {@code cr.getBinary()}'s buffer is not guaranteed to survive past the next read. */
    private byte[] uuidBytes(ColumnReader cr) {
        ByteBuffer bb = cr.getBinary().toByteBuffer();
        byte[] raw = new byte[16];
        bb.get(raw);
        return raw;
    }

    /**
     * Reconstructs a DECIMAL column's current value as a {@link BigDecimal}. Correctness-first, not
     * bulk-decoded (unlike the fixed-width numeric kinds -- see {@link #isBulkKind()}): every physical
     * source Parquet allows for DECIMAL needs its own reassembly (a plain scaled long for INT32/INT64,
     * a big-endian two's-complement byte array -- per parquet-format's LogicalTypes.md -- for BINARY/
     * FIXED_LEN_BYTE_ARRAY), and {@link DecimalVector#set(int, BigDecimal)} is the one Arrow API that
     * accepts all three uniformly without this class having to hand-roll 128-bit arithmetic.
     */
    private BigDecimal decimalOf(ColumnReader cr) {
        return switch (decimalSource) {
            case INT32 -> BigDecimal.valueOf(cr.getInteger(), decimalScale);
            case INT64 -> BigDecimal.valueOf(cr.getLong(), decimalScale);
            case BYTES -> {
                ByteBuffer bb = cr.getBinary().toByteBuffer();
                byte[] raw = new byte[bb.remaining()];
                bb.get(raw);
                yield new BigDecimal(new BigInteger(raw), decimalScale); // BigInteger(byte[]) is big-endian two's complement -- matches the spec exactly
            }
        };
    }

    private boolean isBulkKind() {
        switch (kind) {
            case INT32: case INT8: case INT16: case DATE_DAY: case INT64: case TIMESTAMP: case FLOAT: case DOUBLE:
            case UINT8: case UINT16: case UINT32: case UINT64: // same physical decode as their signed counterparts
                return true;
            default:
                return false;
        }
    }

    /** Drains {@code n} entries of a flat numeric column a page segment at a time; see the class Javadoc. */
    private void fillBulk(FastColumnCursor fc, int n, FieldVector v) {
        int at = 0;
        while (at < n) {
            final int m = fc.beginBulk(n - at);
            writeSegment(fc, m, at, v);
            fc.endBulk();
            at += m;
        }
    }

    private void writeSegment(FastColumnCursor fc, int m, int at, FieldVector v) {
        final int maxDef = this.maxDef;
        final int base = fc.bulkBase();
        final boolean required = maxDef == 0;
        final int[] defs = fc.bulkDefs();
        final int d0 = fc.bulkDefBase();
        switch (kind) {
            case INT32: {
                final IntVector o = (IntVector) v;
                final int[] a = fc.bulkInts();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            case DATE_DAY: {
                final DateDayVector o = (DateDayVector) v;
                final int[] a = fc.bulkInts();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            case INT16: {
                final SmallIntVector o = (SmallIntVector) v;
                final int[] a = fc.bulkInts();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, (short) a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, (short) a[j++]); }
                break;
            }
            case INT8: {
                final TinyIntVector o = (TinyIntVector) v;
                final int[] a = fc.bulkInts();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, (byte) a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, (byte) a[j++]); }
                break;
            }
            case INT64: {
                final BigIntVector o = (BigIntVector) v;
                final long[] a = fc.bulkLongs();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            case TIMESTAMP: {
                final TimeStampVector o = (TimeStampVector) v;
                final long[] a = fc.bulkLongs();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            case FLOAT: {
                final Float4Vector o = (Float4Vector) v;
                final float[] a = fc.bulkFloats();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            case DOUBLE: {
                final Float8Vector o = (Float8Vector) v;
                final double[] a = fc.bulkDoubles();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            case UINT32: {
                final UInt4Vector o = (UInt4Vector) v;
                final int[] a = fc.bulkInts();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            case UINT16: {
                final UInt2Vector o = (UInt2Vector) v;
                final int[] a = fc.bulkInts();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, (short) a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, (short) a[j++]); }
                break;
            }
            case UINT8: {
                final UInt1Vector o = (UInt1Vector) v;
                final int[] a = fc.bulkInts();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, (byte) a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, (byte) a[j++]); }
                break;
            }
            case UINT64: {
                final UInt8Vector o = (UInt8Vector) v;
                final long[] a = fc.bulkLongs();
                if (required) { for (int k = 0; k < m; k++) o.set(at + k, a[base + k]); }
                else { int j = base; for (int k = 0; k < m; k++) if (defs[d0 + k] == maxDef) o.set(at + k, a[j++]); }
                break;
            }
            default:
                throw new IllegalStateException("not a bulk kind: " + kind);
        }
    }

    /**
     * Writes the current value of {@code cr} at index {@code i}, growing the vector if needed. Used for
     * nested leaves whose slot count is unknown until the levels have been read. The caller has already
     * established that the value is present (definition level == max); nulls are simply never written.
     */
    public void writeSafe(ColumnReader cr, int i, FieldVector v) {
        switch (kind) {
            case INT32: ((IntVector) v).setSafe(i, cr.getInteger()); break;
            case INT64: ((BigIntVector) v).setSafe(i, cr.getLong()); break;
            case DOUBLE: ((Float8Vector) v).setSafe(i, cr.getDouble()); break;
            case FLOAT: ((Float4Vector) v).setSafe(i, cr.getFloat()); break;
            case BOOL: ((BitVector) v).setSafe(i, cr.getBoolean() ? 1 : 0); break;
            case INT8: ((TinyIntVector) v).setSafe(i, (byte) cr.getInteger()); break;
            case INT16: ((SmallIntVector) v).setSafe(i, (short) cr.getInteger()); break;
            case DATE_DAY: ((DateDayVector) v).setSafe(i, cr.getInteger()); break;
            case TIMESTAMP: ((TimeStampVector) v).setSafe(i, cr.getLong()); break;
            case UTF8: {
                ByteBuffer bb = cr.getBinary().toByteBuffer();
                ((VarCharVector) v).setSafe(i, bb, bb.position(), bb.remaining());
                break;
            }
            case BINARY: {
                ByteBuffer bb = cr.getBinary().toByteBuffer();
                ((VarBinaryVector) v).setSafe(i, bb, bb.position(), bb.remaining());
                break;
            }
            case DECIMAL: {
                DecimalVector dv = (DecimalVector) v;
                // DecimalVector has no setSafe(int, BigDecimal) overload (unlike the fixed-width numeric
                // vectors above) -- grow capacity ourselves first, exactly what setSafe would do internally.
                while (dv.getValueCapacity() <= i) dv.reAlloc();
                dv.set(i, decimalOf(cr));
                break;
            }
            case UINT8: ((UInt1Vector) v).setSafe(i, (byte) cr.getInteger()); break;
            case UINT16: ((UInt2Vector) v).setSafe(i, (short) cr.getInteger()); break;
            case UINT32: ((UInt4Vector) v).setSafe(i, cr.getInteger()); break;
            case UINT64: ((UInt8Vector) v).setSafe(i, cr.getLong()); break;
            case UUID: {
                FixedSizeBinaryVector fv = (FixedSizeBinaryVector) v;
                while (fv.getValueCapacity() <= i) fv.reAlloc(); // FixedSizeBinaryVector.setSafe(int,byte[]) does not exist either -- same pattern as DECIMAL above
                fv.set(i, uuidBytes(cr));
                break;
            }
            default: throw new IllegalStateException("unhandled kind " + kind);
        }
    }
}
