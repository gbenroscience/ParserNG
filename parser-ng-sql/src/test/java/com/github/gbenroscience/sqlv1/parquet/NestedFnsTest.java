package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
import com.github.gbenroscience.sqlv1.SqlParser;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Path resolution against an Arrow schema with a struct, a list, a map and a list of structs. */
class NestedFnsTest {

    private static Field f(String n, ArrowType t, Field... kids) {
        return new Field(n, FieldType.nullable(t), kids.length == 0 ? null : List.of(kids));
    }

    private static final ArrowType INT32 = new ArrowType.Int(32, true);

    private static final Schema ARROW = new Schema(List.of(
            f("id", new ArrowType.Int(64, true)),
            f("s", ArrowType.Struct.INSTANCE, f("n", INT32), f("label", ArrowType.Utf8.INSTANCE),
                    f("inner", ArrowType.Struct.INSTANCE, f("deep", INT32))),
            f("nums", ArrowType.List.INSTANCE, f("element", INT32)),
            f("m", new ArrowType.Map(false), new Field("entries", FieldType.notNullable(ArrowType.Struct.INSTANCE),
                    List.of(new Field("key", FieldType.notNullable(ArrowType.Utf8.INSTANCE), null), f("value", INT32)))),
            f("items", ArrowType.List.INSTANCE, f("element", ArrowType.Struct.INSTANCE, f("price", INT32), f("name", ArrowType.Utf8.INSTANCE)))));

    private static final TableSchema FILE = TableSchema.of(ARROW);

    private static NestedFns.Result rw(String sql) {
        return NestedFns.rewrite(SqlParser.parse(sql), FILE, ARROW, 0);
    }

    private static void assertNamed(String sql, String fragment) {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> rw(sql), sql);
        assertTrue(e.getMessage().contains(fragment), sql + " -> " + e.getMessage());
    }

    @Test
    void pathsAreRecognisedAndTopLevelColumnsFound() {
        assertTrue(NestedFns.isPath("s.n", FILE));
        assertTrue(NestedFns.isPath("nums[0]", FILE));
        assertFalse(NestedFns.isPath("id", FILE));
        assertFalse(NestedFns.isPath("id.x", FILE), "id is not nested");
        assertFalse(NestedFns.isPath("zzz.x", FILE));
        assertEquals("s", NestedFns.topLevel("s.items[2].price"));
        assertEquals("m", NestedFns.topLevel("m['k']"));
        assertEquals("id", NestedFns.topLevel("id"));
    }

    @Test
    void leavesBecomeDerivedColumnsOfTheirOwnType() {
        var r = rw("SELECT s.n, s.label, s.inner.deep, nums[1], m['a'], items[0].price, items[1].name FROM t");
        assertEquals(7, r.columns().size());
        assertEquals(ColumnKind.INT32, r.columns().get(0).kind);
        assertEquals(ColumnKind.STRING, r.columns().get(1).kind);
        assertEquals(ColumnKind.INT32, r.columns().get(2).kind);
        assertEquals(ColumnKind.INT32, r.columns().get(3).kind);
        assertEquals(ColumnKind.INT32, r.columns().get(4).kind);
        assertEquals(ColumnKind.INT32, r.columns().get(5).kind);
        assertEquals(ColumnKind.STRING, r.columns().get(6).kind);
        assertEquals("s.n", r.stmt().items().get(0).alias(), "the path names the output");
        assertEquals(r.columns().get(0).name, r.stmt().items().get(0).exprText());
    }

    @Test
    void theSamePathSharesOneColumnAcrossClauses() {
        var r = rw("SELECT s.n FROM t WHERE s.n > 3 ORDER BY s.n");
        assertEquals(1, r.columns().size());
    }

    @Test
    void cardinalityOfAListOrMapIsANumericMask() {
        var r = rw("SELECT CARDINALITY(nums), CARDINALITY(m) FROM t WHERE CARDINALITY(nums) > 1");
        assertEquals(2, r.masks().size());
        assertTrue(r.columns().isEmpty());
        assertEquals("CARDINALITY(nums)", r.stmt().items().get(0).alias());
    }

    @Test
    void schemaClassifiesAPathBeforeItIsRewritten() {
        assertEquals(ColumnKind.INT32, FILE.kind(QuotedIdentifiers.encode("s.n")));
        assertEquals(ColumnKind.STRING, FILE.kind(QuotedIdentifiers.encode("s.label")));
        assertNull(FILE.kind(QuotedIdentifiers.encode("s.nope")));
        assertFalse(FILE.hasFileColumn(QuotedIdentifiers.encode("s.n")), "a path is not a file column");
    }

    @Test
    void unresolvablePathsFailWithNamedErrors() {
        assertNamed("SELECT s.nope FROM t", "no field 'nope'");
        assertNamed("SELECT s.nope FROM t", "fields:");
        assertNamed("SELECT s FROM t WHERE s.n.x = 1", "not a struct");
        assertNamed("SELECT nums.x FROM t", "not a struct");
        assertNamed("SELECT s[0] FROM t", "cannot be indexed");
        assertNamed("SELECT nums['k'] FROM t", "by position");
        assertNamed("SELECT m[0] FROM t", "keys are strings");
        assertNamed("SELECT items[0] FROM t", "not a value");
        assertNamed("SELECT s.inner FROM t", "not a value");
        assertNamed("SELECT CARDINALITY(s.n) FROM t", "needs a list or map");
        assertNamed("SELECT CARDINALITY(s) FROM t", "needs a list or map");
    }

    @Test
    void statementsWithoutPathsAreUntouched() {
        var st = SqlParser.parse("SELECT id FROM t WHERE id > 1");
        assertSame(st, NestedFns.rewrite(st, FILE, ARROW, 0).stmt());
    }
}
