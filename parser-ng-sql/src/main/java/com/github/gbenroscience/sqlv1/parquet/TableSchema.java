package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
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
    /** Derived columns computed by the bridge per batch (string function results); never listed by {@link #names()}. */
    private final Map<String, ColumnKind> virtual = new LinkedHashMap<>();
    /** The file's Arrow schema when known, so a nested-path reference can be classified; else null. */
    private Schema arrow;
    private final Map<String, ArrowType> virtualTypes = new LinkedHashMap<>();

    private TableSchema() { }

    static TableSchema of(Schema schema) {
        TableSchema t = new TableSchema();
        t.arrow = schema;
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
        ColumnKind k = virtual.get(column);
        if (k != null) return k;
        String real = QuotedIdentifiers.real(column);
        k = kinds.get(real);
        if (k == null && arrow != null && !real.equals(column) && NestedFns.isPath(real, this)) {
            org.apache.arrow.vector.types.pojo.Field leaf = NestedFns.leafField(real, this, arrow);
            return leaf == null ? null : kindOf(leaf.getType());
        }
        return k;
    }

    /** The Arrow field a nested-path reference ends at, or null if {@code ref} is not a resolvable path. */
    org.apache.arrow.vector.types.pojo.Field pathField(String ref) {
        String real = QuotedIdentifiers.real(ref);
        if (arrow == null || real.equals(ref) || !NestedFns.isPath(real, this)) return null;
        return NestedFns.leafField(real, this, arrow);
    }

    /** @return {@code {precision, scale}} of a DECIMAL column, or {@code null} for any other column */
    int[] decimal(String column) {
        return decimals.get(QuotedIdentifiers.real(column));
    }

    ArrowType type(String column) {
        ArrowType v = virtualTypes.get(column);
        return v != null ? v : types.get(QuotedIdentifiers.real(column));
    }

    /** @return top-level column names in file order */
    java.util.List<String> names() {
        return new java.util.ArrayList<>(kinds.keySet());
    }

    /** True for a file column (a quoted-identifier reference counts as its real name) or a derived column. */
    boolean has(String column) {
        return virtual.containsKey(column) || kinds.containsKey(QuotedIdentifiers.real(column));
    }

    /** True only for a column of the file itself. */
    boolean hasFileColumn(String column) {
        return kinds.containsKey(QuotedIdentifiers.real(column));
    }

    /** True for a derived column. */
    boolean isVirtual(String column) {
        return virtual.containsKey(column);
    }

    /** A copy that also knows the derived columns {@code extra} (name, kind and Arrow type). */
    TableSchema withVirtual(java.util.List<DerivedColumns.Column> extra) {
        if (extra.isEmpty()) return this;
        TableSchema t = new TableSchema();
        t.kinds.putAll(kinds);
        t.types.putAll(types);
        t.decimals.putAll(decimals);
        t.virtual.putAll(virtual);
        t.virtualTypes.putAll(virtualTypes);
        t.arrow = arrow;
        for (DerivedColumns.Column c : extra) {
            t.virtual.put(c.name, c.kind);
            if (c.type != null) t.virtualTypes.put(c.name, c.type);
        }
        return t;
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
