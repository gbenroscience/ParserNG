package com.github.gbenroscience.sqlv1.parquet;

import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Column kinds of one Parquet file, keyed by top-level column name.
 * Built from the Arrow schema of a scan, so it matches the vectors the
 * bridge will actually see.
 */
final class TableSchema {

    private final Map<String, ColumnKind> kinds = new LinkedHashMap<>();
    private final Map<String, ArrowType> types = new LinkedHashMap<>();
    private final Map<String, int[]> decimals = new LinkedHashMap<>();

    private TableSchema() { }

    static TableSchema of(Schema schema) {
        TableSchema t = new TableSchema();
        for (Field f : schema.getFields()) {
            t.kinds.put(f.getName(), kindOf(f.getType()));
            t.types.put(f.getName(), f.getType());
            if (f.getType() instanceof ArrowType.Decimal d) t.decimals.put(f.getName(), new int[] {d.getPrecision(), d.getScale()});
        }
        return t;
    }

    static ColumnKind kindOf(ArrowType a) {
        if (a instanceof ArrowType.Int i) {
            int w = i.getBitWidth();
            boolean s = i.getIsSigned();
            return switch (w) {
                case 8 -> s ? ColumnKind.INT8 : ColumnKind.UINT8;
                case 16 -> s ? ColumnKind.INT16 : ColumnKind.UINT16;
                case 32 -> s ? ColumnKind.INT32 : ColumnKind.UINT32;
                case 64 -> s ? ColumnKind.INT64 : ColumnKind.UINT64;
                default -> ColumnKind.OTHER;
            };
        }
        if (a instanceof ArrowType.FloatingPoint f) {
            FloatingPointPrecision p = f.getPrecision();
            return p == FloatingPointPrecision.SINGLE ? ColumnKind.FLOAT32
                    : p == FloatingPointPrecision.DOUBLE ? ColumnKind.FLOAT64 : ColumnKind.OTHER;
        }
        if (a instanceof ArrowType.Decimal) return ColumnKind.DECIMAL;
        if (a instanceof ArrowType.Bool) return ColumnKind.BOOL;
        if (a instanceof ArrowType.Utf8) return ColumnKind.STRING;
        if (a instanceof ArrowType.Binary) return ColumnKind.BINARY;
        if (a instanceof ArrowType.Date) return ColumnKind.DATE;
        if (a instanceof ArrowType.Timestamp) return ColumnKind.TIMESTAMP;
        return ColumnKind.OTHER;
    }

    /** @return the kind, or {@code null} if the file has no such top-level column */
    ColumnKind kind(String column) {
        return kinds.get(column);
    }

    /** @return {@code {precision, scale}} of a DECIMAL column, or {@code null} for any other column */
    int[] decimal(String column) {
        return decimals.get(column);
    }

    ArrowType type(String column) {
        return types.get(column);
    }

    /** @return top-level column names in file order */
    java.util.List<String> names() {
        return new java.util.ArrayList<>(kinds.keySet());
    }

    boolean has(String column) {
        return kinds.containsKey(column);
    }

    /** Test factory: kinds plus {@code {precision, scale}} for the named DECIMAL columns. */
    static TableSchema ofKinds(Map<String, ColumnKind> kinds, Map<String, int[]> decimals) {
        TableSchema t = ofKinds(kinds);
        t.decimals.putAll(decimals);
        return t;
    }

    /** Test/planning factory: kinds only, no Arrow types. */
    static TableSchema ofKinds(Map<String, ColumnKind> kinds) {
        TableSchema t = new TableSchema();
        t.kinds.putAll(kinds);
        return t;
    }
}
