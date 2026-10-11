package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Rewriting of COALESCE/NULLIF into derived columns, and the kernels that fill them. */
class ScalarFnsTest {

    private static final TableSchema S = TableSchema.ofKinds(Map.of(
            "a", ColumnKind.INT32, "b", ColumnKind.FLOAT64, "c", ColumnKind.INT64, "name", ColumnKind.STRING));

    private static ScalarFns.Result rw(String sql) {
        return ScalarFns.rewrite(SqlParser.parse(sql), S);
    }

    private static Double[] run(ScalarFns.Result r, int mask, Double[] a, Double[] b, Double[] c) {
        try (RootAllocator alloc = new RootAllocator();
             Float8Vector va = vec(alloc, "a", a);
             Float8Vector vb = vec(alloc, "b", b);
             Float8Vector vc = vec(alloc, "c", c);
             Float8Vector out = new Float8Vector("out", alloc)) {
            out.allocateNew(Math.max(a.length, 1));
            VectorSchemaRoot widened = new VectorSchemaRoot(java.util.List.of(va, vb, vc));
            r.masks().get(mask).eval(widened, a.length, out);
            Double[] res = new Double[a.length];
            for (int i = 0; i < a.length; i++) res[i] = out.isNull(i) ? null : out.get(i);
            return res;
        }
    }

    private static Float8Vector vec(RootAllocator alloc, String name, Double[] v) {
        Float8Vector x = new Float8Vector(name, alloc);
        x.allocateNew(Math.max(v.length, 1));
        for (int i = 0; i < v.length; i++) if (v[i] == null) x.setNull(i); else x.set(i, v[i]);
        x.setValueCount(v.length);
        return x;
    }

    private static final Double N = null;

    @Test
    void coalesceReplacesTheCallAndKeepsTheOutputName() {
        var r = rw("SELECT COALESCE(a, 0) + 1 AS x, COALESCE(a, b) FROM t");
        assertEquals("pngmaska + 1", r.stmt().items().get(0).exprText());
        assertEquals("x", r.stmt().items().get(0).alias());
        assertEquals("COALESCE(a, b)", r.stmt().items().get(1).alias(), "the original text names the column");
        assertEquals(2, r.masks().size());
    }

    @Test
    void coalesceReturnsTheFirstNonNull() {
        var r = rw("SELECT COALESCE(a, b, c, 7) FROM t");
        assertArrayEquals(new Double[] {1.0, 2.0, 3.0, 7.0},
                run(r, 0, new Double[] {1.0, N, N, N}, new Double[] {5.0, 2.0, N, N}, new Double[] {5.0, 5.0, 3.0, N}));
        assertEquals(java.util.List.of("a", "b", "c"), r.masks().get(0).widenedColumns());
        assertArrayEquals(new Double[] {N}, run(rw("SELECT COALESCE(a, b) FROM t"), 0, new Double[] {N}, new Double[] {N}, new Double[] {N}));
    }

    @Test
    void nullifIsNullWhenEqualOrWhenTheFirstIsNull() {
        var r = rw("SELECT NULLIF(a, 0) FROM t");
        assertArrayEquals(new Double[] {N, 5.0, N, N},
                run(r, 0, new Double[] {0.0, 5.0, N, -0.0}, new Double[4], new Double[4]));
        r = rw("SELECT NULLIF(a, b) FROM t");
        assertArrayEquals(new Double[] {N, 1.0, 1.0},
                run(r, 0, new Double[] {2.0, 1.0, 1.0}, new Double[] {2.0, 2.0, N}, new Double[3]),
                "a null second argument never equals the first");
    }

    @Test
    void callsNestAndRepeatedCallsShareOneColumn() {
        var r = rw("SELECT COALESCE(NULLIF(a, 0), b) FROM t");
        assertEquals(1, r.masks().size());
        assertArrayEquals(new Double[] {4.0, 3.0}, run(r, 0, new Double[] {0.0, 3.0}, new Double[] {4.0, 4.0}, new Double[2]));
        r = rw("SELECT a FROM t WHERE COALESCE(a, 0) > 5 AND NULLIF(b, -1) < 3 ORDER BY COALESCE(a, 0) DESC");
        assertEquals(2, r.masks().size());
        assertEquals("pngmaska", r.stmt().orderBy().get(0).exprText());
    }

    @Test
    void everyClauseIsRewritten() {
        var r = rw("SELECT SUM(COALESCE(a, 0)) FROM t GROUP BY COALESCE(b, 1)");
        assertEquals("pngmaska", r.stmt().items().get(0).aggregate().argExprText());
        assertEquals("SUM(COALESCE(a, 0))", r.stmt().items().get(0).alias());
        assertEquals("pngmaskb", r.stmt().groupBy().get(0));
    }

    @Test
    void statementsWithoutACallAreReturnedUnchanged() {
        var st = SqlParser.parse("SELECT a FROM t WHERE a > 1 LIMIT 3 OFFSET 2");
        var r = ScalarFns.rewrite(st, S);
        assertSame(st, r.stmt());
        assertTrue(r.masks().isEmpty());
        assertEquals(2, rw("SELECT COALESCE(a, 0) FROM t LIMIT 3 OFFSET 2").stmt().offset());
        assertTrue(rw("SELECT name FROM t WHERE name = 'COALESCE(a, 0)'").masks().isEmpty(), "inside a string literal");
        assertTrue(ScalarFns.containsCall("1 + coalesce (a, 2)"));
        assertFalse(ScalarFns.containsCall("coalesced + 1"));
    }

    @Test
    void unsupportedArgumentsFailWithNamedErrors() {
        assertTrue(assertThrows(ParquetSqlException.class, () -> rw("SELECT COALESCE(name, 'x') FROM t")).getMessage().contains("STRING"));
        assertTrue(assertThrows(ParquetSqlException.class, () -> rw("SELECT COALESCE(a + b, 0) FROM t")).getMessage().contains("not a numeric column"));
        assertTrue(assertThrows(ParquetSqlException.class, () -> rw("SELECT NULLIF(a) FROM t")).getMessage().contains("exactly 2"));
        assertTrue(assertThrows(ParquetSqlException.class, () -> rw("SELECT a FROM t WHERE COALESCE(a, b) IS NULL")).getMessage().contains("IS [NOT] NULL"));
        assertTrue(assertThrows(ParquetSqlException.class, () -> rw("SELECT a FROM t WHERE COALESCE(a, b) LIKE 'x'")).getMessage().contains("LIKE"));
    }
}
