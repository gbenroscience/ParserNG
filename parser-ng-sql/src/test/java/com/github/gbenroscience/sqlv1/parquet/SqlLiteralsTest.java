package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.CompOp;
import com.github.gbenroscience.sqlv1.parquet.SqlLiterals.Kind;
import com.github.gbenroscience.sqlv1.parquet.SqlLiterals.LongTest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Literal classification and the exact temporal comparison arithmetic. */
class SqlLiteralsTest {

    private static SqlLiterals.Operand op(String t) {
        return SqlLiterals.operand(t, c -> c.equals("col") || c.equals("day"));
    }

    @Test
    void classifiesOperands() {
        assertEquals(Kind.COLUMN, op("col").kind());
        assertEquals(Kind.STRING, op("'it''s'").kind());
        assertEquals("it's", op("'it''s'").text());
        assertEquals(Kind.BOOL, op("true").kind());
        assertTrue(op("TRUE").bool());
        assertFalse(op("False").bool());
        assertEquals(Kind.NULL, op("null").kind());
        assertEquals(Kind.NUMBER, op("- 5").kind());
        assertEquals(Kind.NUMBER, op("1.5").kind());
        assertEquals(Kind.DATE, op("DATE '2024-01-31'").kind());
        assertEquals("2024-01-31", op("date '2024-01-31'").text());
        assertEquals(Kind.TIMESTAMP, op("TIMESTAMP '2024-01-31 12:00:00'").kind());
        assertEquals(Kind.OTHER, op("x * 2").kind());
        assertEquals(Kind.OTHER, op("unknown_col").kind(), "an identifier that is not a column");
    }

    @Test
    void dateLiteral() {
        assertEquals(0L, SqlLiterals.epochDay("1970-01-01"));
        assertEquals(19_723L, SqlLiterals.epochDay("2024-01-01"));
        assertThrows(ParquetSqlException.class, () -> SqlLiterals.epochDay("2024-13-01"));
        assertThrows(ParquetSqlException.class, () -> SqlLiterals.epochDay("yesterday"));
    }

    @Test
    void timestampLiteralForms() {
        var a = SqlLiterals.timestamp("2024-01-01 12:00:00");
        var b = SqlLiterals.timestamp("2024-01-01T12:00:00");
        assertEquals(a, b);
        assertFalse(a.hasZone());
        assertEquals(1_704_110_400L, a.epochSecond());
        var z = SqlLiterals.timestamp("2024-01-01 12:00:00Z");
        assertTrue(z.hasZone());
        assertEquals(1_704_110_400L, z.epochSecond());
        assertEquals(1_704_110_400L - 7200, SqlLiterals.timestamp("2024-01-01 12:00:00+02:00").epochSecond());
        assertEquals(500_000_000, SqlLiterals.timestamp("2024-01-01 12:00:00.5").nano());
        assertEquals(1_704_067_200L, SqlLiterals.timestamp("2024-01-01").epochSecond(), "bare date is midnight");
        assertThrows(ParquetSqlException.class, () -> SqlLiterals.timestamp("nope"));
    }

    @Test
    void zonedLiteralAgainstNaiveColumnIsRejected() {
        var lit = SqlLiterals.timestamp("2024-01-01 12:00:00Z");
        ParquetSqlException e = assertThrows(ParquetSqlException.class,
                () -> SqlLiterals.timestampTest("ts", false, CompOp.GT, lit, 1_000_000L));
        assertTrue(e.getMessage().contains("'ts'") && e.getMessage().contains("ambiguous"));
        assertNotNull(SqlLiterals.timestampTest("ts", true, CompOp.GT, lit, 1_000_000L));
    }

    @Test
    void timestampComparesExactlyInEveryUnit() {
        var lit = SqlLiterals.timestamp("2024-01-01 12:00:00");
        long micros = 1_704_110_400L * 1_000_000L;
        LongTest ge = SqlLiterals.timestampTest("ts", true, CompOp.GE, lit, 1_000_000L);
        assertTrue(ge.test(micros));
        assertTrue(ge.test(micros + 1));
        assertFalse(ge.test(micros - 1));
        LongTest millis = SqlLiterals.timestampTest("ts", true, CompOp.EQ, lit, 1_000L);
        assertTrue(millis.test(1_704_110_400_000L));
        LongTest nanos = SqlLiterals.timestampTest("ts", true, CompOp.LT, lit, 1_000_000_000L);
        assertTrue(nanos.test(1_704_110_400L * 1_000_000_000L - 1));
        assertFalse(nanos.test(1_704_110_400L * 1_000_000_000L));
    }

    @Test
    void literalFinerThanTheColumnUnitNeverMisclassifies() {
        // 12:00:00.0005 is half a millisecond after the tick t = ...400_000 ms.
        var lit = SqlLiterals.timestamp("2024-01-01 12:00:00.0005");
        long t = 1_704_110_400_000L;
        assertFalse(SqlLiterals.timestampTest("ts", true, CompOp.EQ, lit, 1_000L).test(t), "no millisecond equals it");
        assertTrue(SqlLiterals.timestampTest("ts", true, CompOp.NEQ, lit, 1_000L).test(t));
        assertFalse(SqlLiterals.timestampTest("ts", true, CompOp.GT, lit, 1_000L).test(t), "t < literal");
        assertTrue(SqlLiterals.timestampTest("ts", true, CompOp.GT, lit, 1_000L).test(t + 1), "t+1ms > literal");
        assertTrue(SqlLiterals.timestampTest("ts", true, CompOp.LT, lit, 1_000L).test(t));
        assertFalse(SqlLiterals.timestampTest("ts", true, CompOp.LT, lit, 1_000L).test(t + 1));
        assertFalse(SqlLiterals.timestampTest("ts", true, CompOp.GE, lit, 1_000L).test(t));
        assertTrue(SqlLiterals.timestampTest("ts", true, CompOp.GE, lit, 1_000L).test(t + 1));
        assertTrue(SqlLiterals.timestampTest("ts", true, CompOp.LE, lit, 1_000L).test(t));
        assertFalse(SqlLiterals.timestampTest("ts", true, CompOp.LE, lit, 1_000L).test(t + 1));
    }

    @Test
    void negativeEpochRoundsTowardMinusInfinity() {
        var lit = SqlLiterals.timestamp("1969-12-31 23:59:59.9995"); // 0.5 ms before the epoch
        assertTrue(SqlLiterals.timestampTest("ts", true, CompOp.LT, lit, 1_000L).test(-1), "-1 ms < -0.5 ms");
        assertFalse(SqlLiterals.timestampTest("ts", true, CompOp.LT, lit, 1_000L).test(0));
        assertTrue(SqlLiterals.timestampTest("ts", true, CompOp.GT, lit, 1_000L).test(0));
    }

    @Test
    void literalBeyondNanosecondRangeIsAConstant() {
        var far = SqlLiterals.timestamp("9999-12-31 00:00:00");
        LongTest lt = SqlLiterals.timestampTest("ts", true, CompOp.LT, far, 1_000_000_000L); // overflows int64 nanos
        assertEquals(1, lt.constant());
        LongTest gt = SqlLiterals.timestampTest("ts", true, CompOp.GT, far, 1_000_000_000L);
        assertEquals(0, gt.constant());
    }

    @Test
    void dateAgainstTimestampColumnIsMidnight() {
        LongTest t = SqlLiterals.dateAsTimestampTest("ts", true, CompOp.GE, 19_723L, 1_000_000L);
        assertTrue(t.test(19_723L * 86_400_000_000L));
        assertFalse(t.test(19_723L * 86_400_000_000L - 1));
    }
}
