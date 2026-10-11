package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Numeric CAST: arithmetic for numbers, a parsing column for strings. */
class CastFnsTest {

    private static final TableSchema S = TableSchema.ofKinds(Map.of(
            "x", ColumnKind.FLOAT64, "n", ColumnKind.INT32, "name", ColumnKind.STRING, "ts", ColumnKind.TIMESTAMP));

    private static CastFns.Result rw(String sql) {
        return CastFns.rewrite(SqlParser.parse(sql, true), S, 0);
    }

    @Test
    void aCastOfANumberIsTheSameArithmeticAsTheDesugaredParser() {
        var r = rw("SELECT CAST(x AS INTEGER), CAST(x AS DOUBLE), CAST(n AS BIGINT), CAST(x AS DECIMAL) FROM t");
        assertTrue(r.masks().isEmpty());
        assertEquals("((x) - ((x) % 1))", r.stmt().items().get(0).exprText());
        assertEquals("(x)", r.stmt().items().get(1).exprText());
        assertEquals("((n) - ((n) % 1))", r.stmt().items().get(2).exprText());
        assertEquals("(x)", r.stmt().items().get(3).exprText());
        assertEquals("CAST(x AS INTEGER)", r.stmt().items().get(0).alias(), "the original text names the column");
    }

    @Test
    void aCastOfAStringBecomesAParsingColumnOnePerTargetKind() {
        var r = rw("SELECT CAST(name AS INTEGER), CAST(name AS INTEGER) + 1, CAST(name AS DOUBLE), CAST(name AS BIGINT) FROM t WHERE CAST(name AS INTEGER) > 5");
        assertEquals(3, r.masks().size(), "INTEGER is shared by three uses; DOUBLE and BIGINT have their own");
        assertEquals(r.masks().get(0).name, r.stmt().items().get(0).exprText());
        assertEquals(r.masks().get(0).name + " + 1", r.stmt().items().get(1).exprText());
    }

    @Test
    void castsNestInsideEachOtherAndInsideOtherCalls() {
        var r = rw("SELECT CAST(CAST(x AS INTEGER) AS DOUBLE), SUM(CAST(x AS INTEGER)) FROM t");
        assertTrue(r.masks().isEmpty());
        assertEquals("(((x) - ((x) % 1)))", r.stmt().items().get(0).exprText());
        assertEquals("((x) - ((x) % 1))", r.stmt().items().get(1).aggregate().argExprText());
    }

    @Test
    void stringFunctionsAndLiteralsCountAsStrings() {
        assertEquals(1, rw("SELECT CAST(SUBSTRING(name, 2, 3) AS INTEGER) FROM t").masks().size());
        assertEquals(1, rw("SELECT CAST('12' AS INTEGER) FROM t").masks().size());
        assertEquals(1, rw("SELECT CAST(name || '0' AS DOUBLE) FROM t").masks().size());
    }

    @Test
    void otherTargetsAreLeftForTheirOwnRewriters() {
        var r = rw("SELECT CAST(ts AS DATE), CAST(n AS VARCHAR), CAST(x AS DECIMAL(9,2)) FROM t");
        assertTrue(r.masks().isEmpty());
        assertEquals("CAST(ts AS DATE)", r.stmt().items().get(0).exprText());
        assertEquals("CAST(n AS VARCHAR)", r.stmt().items().get(1).exprText());
        assertEquals("CAST(x AS DECIMAL(9,2))", r.stmt().items().get(2).exprText());
    }

    @Test
    void statementsWithoutANumericCastAreUntouched() {
        var st = SqlParser.parse("SELECT n FROM t WHERE n > 1", true);
        assertSame(st, CastFns.rewrite(st, S, 0).stmt());
    }
}
