package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScanException;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.LogicalTypeAnnotation.*;
import org.apache.parquet.schema.PrimitiveType;

import java.nio.ByteBuffer;
import java.nio.file.Path;

/**
 * Per-column mapping Parquet -> Arrow plus the primitive decode loop.
 *
 * <p>Scope of this class (flat, non-repeated primitive columns only): anything else throws a
 * {@link ParquetScanException} naming the column and the reason. Nothing is ever coerced silently.
 *
 * <p>Hot loop: no boxing. One virtual {@code ColumnReader} call per value (parquet-java's public
 * API has no batch decode) plus one direct Arrow {@code set}. A null is left as an unset validity
 * bit; the value slot is never written, so null is never turned into 0/false/"".
 */
public final class ColumnPlan {

    enum Kind { BOOL, INT8, INT16, INT32, DATE_DAY, INT64, TIMESTAMP, FLOAT, DOUBLE, UTF8, BINARY }

    private final Kind kind;
    private final int maxDef;
    private final Field field;

    private ColumnPlan(Kind kind, int maxDef, Field field) {
        this.kind = kind;
        this.maxDef = maxDef;
        this.field = field;
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
                else throw unsupported(file, name, "INT32 with logical type " + lt + " (unsigned ints, DECIMAL, TIME not supported yet)");
                break;
            case INT64:
                if (lt == null || (lt instanceof IntLogicalTypeAnnotation i && i.isSigned() && i.getBitWidth() == 64)) {
                    k = Kind.INT64; at = new ArrowType.Int(64, true);
                } else if (lt instanceof TimestampLogicalTypeAnnotation ts) {
                    k = Kind.TIMESTAMP;
                    org.apache.arrow.vector.types.TimeUnit u;
                    switch (ts.getUnit()) {
                        case MILLIS: u = org.apache.arrow.vector.types.TimeUnit.MILLISECOND; break;
                        case MICROS: u = org.apache.arrow.vector.types.TimeUnit.MICROSECOND; break;
                        default: u = org.apache.arrow.vector.types.TimeUnit.NANOSECOND; break;
                    }
                    at = new ArrowType.Timestamp(u, ts.isAdjustedToUTC() ? "UTC" : null);
                } else throw unsupported(file, name, "INT64 with logical type " + lt + " (unsigned ints, DECIMAL, TIME not supported yet)");
                break;
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
                else throw unsupported(file, name, "BINARY with logical type " + lt + " (DECIMAL, BSON not supported yet)");
                break;
            default:
                throw unsupported(file, name, pt.getPrimitiveTypeName() + " (INT96 / FIXED_LEN_BYTE_ARRAY / UUID / DECIMAL not supported yet)");
        }
        return new ColumnPlan(k, maxDef, new Field(name, new FieldType(nullable, at, null), null));
    }

    private static void needNoLogical(LogicalTypeAnnotation lt, Path file, String name) {
        if (lt != null) throw unsupported(file, name, "logical type " + lt + " on this physical type");
    }

    private static ParquetScanException unsupported(Path file, String col, String why) {
        return new ParquetScanException("Unsupported Parquet column: " + why, file, -1, col, null);
    }

    /** Fills {@code v[0..n)} from {@code cr}; the caller sets value count / row count. */
    public void fill(ColumnReader cr, int n, FieldVector v) {
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
            default:
                throw new IllegalStateException("unhandled kind " + kind);
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
            default: throw new IllegalStateException("unhandled kind " + kind);
        }
    }
}
