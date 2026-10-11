package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.ast.ComparisonExpr;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** The temporal rewriter: literal folding, derived columns, and named errors. */
class TemporalFnsTest {

    private static final TableSchema S = TableSchema.ofKinds(Map.of(
            "ts", ColumnKind.TIMESTAMP, "d", ColumnKind.DATE, "name", ColumnKind.STRING, "id", ColumnKind.INT64, "x", ColumnKind.FLOAT64));

    private static TemporalFns.Result rw(String sql) {
        return TemporalFns.rewrite(SqlParser.parse(sql), S, 0);
    }

    private static String rightOfWhere(TemporalFns.Result r) {
        return ((ComparisonExpr) r.stmt().where()).right();
    }

    private static void assertNamed(String sql, String fragment) {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> rw(sql), sql);
        assertTrue(e.getMessage().contains(fragment), sql + " -> " + e.getMessage());
    }

    @Test
    void literalArithmeticIsFoldedIntoOneLiteralInAPredicate() {
        var r = rw("SELECT id FROM t WHERE ts >= TIMESTAMP '2024-01-31 12:00:00' - INTERVAL '1 month'");
        assertEquals("TIMESTAMP '2023-12-31 12:00:00'", rightOfWhere(r));
        assertTrue(r.columns().isEmpty());
        assertEquals("DATE '2024-02-29'", rightOfWhere(rw("SELECT id FROM t WHERE d = DATE '2024-01-31' + INTERVAL '1 month'")));
        assertEquals("DATE '2024-02-01'", rightOfWhere(rw("SELECT id FROM t WHERE d = CAST('2024-02-01' AS DATE)")));
        assertEquals("TIMESTAMP '2024-02-01 10:30:00.5'", rightOfWhere(rw("SELECT id FROM t WHERE ts = CAST('2024-02-01 10:30:00.5' AS TIMESTAMP)")));
        assertEquals("TIMESTAMP '2024-02-01 06:00:00'", rightOfWhere(rw("SELECT id FROM t WHERE ts = DATE '2024-01-31' + INTERVAL '30 hours'")),
                "a sub-day interval turns a date into a timestamp");
    }

    @Test
    void columnExpressionsBecomeDerivedTypedColumnsAndKeepTheirNames() {
        var r = rw("SELECT ts + INTERVAL '1 day' AS nxt, CAST(ts AS DATE), CAST(d AS TIMESTAMP) FROM t");
        assertEquals(3, r.columns().size());
        assertEquals(ColumnKind.TIMESTAMP, r.columns().get(0).kind);
        assertEquals(ColumnKind.DATE, r.columns().get(1).kind);
        assertEquals(ColumnKind.TIMESTAMP, r.columns().get(2).kind);
        assertEquals("nxt", r.stmt().items().get(0).alias());
        assertEquals("CAST(ts AS DATE)", r.stmt().items().get(1).alias());
        assertEquals(r.columns().get(1).name, r.stmt().items().get(1).exprText());
    }

    @Test
    void aSelectItemIsNeverFoldedEvenWhenItIsAllLiterals() {
        var r = rw("SELECT CAST('2024-02-01' AS DATE) AS c FROM t");
        assertEquals(1, r.columns().size());
        assertEquals(ColumnKind.DATE, r.columns().get(0).kind);
    }

    @Test
    void extractAndDecimalCastAreNumericMasks() {
        var r = rw("SELECT EXTRACT(YEAR FROM ts) AS y, CAST(x AS DECIMAL(10,2)) FROM t WHERE EXTRACT(MONTH FROM d) = 3");
        assertEquals(3, r.masks().size());
        assertTrue(r.columns().isEmpty());
        assertEquals("y", r.stmt().items().get(0).alias());
        assertEquals("CAST(x AS DECIMAL(10,2))", r.stmt().items().get(1).alias());
        assertEquals(r.masks().get(0).name, r.stmt().items().get(0).exprText());
        assertEquals(r.masks().get(2).name, ((ComparisonExpr) r.stmt().where()).left());
        assertEquals(1, rw("SELECT EXTRACT(YEAR FROM ts), EXTRACT(YEAR FROM ts) + 1 FROM t").masks().size(), "the same call shares one column");
    }

    @Test
    void statementsWithoutACallAreUntouched() {
        var st = SqlParser.parse("SELECT id FROM t WHERE id > 1");
        assertSame(st, TemporalFns.rewrite(st, S, 0).stmt());
    }

    @Test
    void unsupportedUsesFailWithNamedErrors() {
        assertNamed("SELECT EXTRACT(fortnight FROM d) FROM t", "supported");
        assertNamed("SELECT EXTRACT(YEAR FROM id) FROM t", "INT64");
        assertNamed("SELECT EXTRACT(YEAR FROM zzz) FROM t", "unknown column");
        assertNamed("SELECT d + INTERVAL '1 month' FROM t", "can only be added to a literal");
        assertNamed("SELECT CAST('2024-02-30' AS DATE) FROM t", "not a valid DATE");
        assertNamed("SELECT id FROM t WHERE d = CAST('2024-13-01' AS DATE)", "not a valid DATE");
        assertNamed("SELECT CAST(name AS DECIMAL(5,2)) FROM t", "STRING");
        assertNamed("SELECT CAST(x AS DECIMAL(2,5)) FROM t", "precision");
        assertNamed("SELECT ts + INTERVAL '3 fortnights' FROM t", "not supported");
        assertNamed("SELECT ts + 1 FROM t WHERE ts = CAST(ts AS DATE) + 1", "INTERVAL");
        assertNamed("SELECT TIMESTAMP '2024-01-01 00:00:00Z' - INTERVAL '1 day' FROM t", "time zone");
    }
}
