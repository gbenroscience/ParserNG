package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.ast.CompOp;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure-logic tests (no Arrow, no files): literal promotion per column kind,
 * and the superset-only soundness rules. Finding 5 of the task: literal typing
 * decides whether parser-ng-parquet pushes a leaf at all.
 */
class PredicateConverterTest {

    private static final TableSchema SCHEMA = TableSchema.ofKinds(Map.of(
            "i32", ColumnKind.INT32, "i64", ColumnKind.INT64, "f32", ColumnKind.FLOAT32,
            "f64", ColumnKind.FLOAT64, "s", ColumnKind.STRING, "d", ColumnKind.DECIMAL,
            "ts", ColumnKind.TIMESTAMP, "u", ColumnKind.UINT32));

    private static Predicate pruning(String where) {
        return PredicateConverter.convert(SqlParser.parse("SELECT i32 FROM t WHERE " + where).where(), SCHEMA);
    }

    @Test
    void int64ColumnGetsLongLiteral() {
        var b = PredicateConverter.bindForTest(ColumnKind.INT64, CompOp.GT, "100");
        assertEquals(Long.class, b.value().getClass());
        assertNotNull(pruning("i64 > 100"));
    }

    @Test
    void int32ColumnGetsIntegerLiteralAndOutOfRangeIsNotPushed() {
        assertEquals(Integer.class, PredicateConverter.bindForTest(ColumnKind.INT32, CompOp.EQ, "7").value().getClass());
        assertNull(pruning("i32 < 5000000000"));
        assertNull(pruning("i32 > 2.5"), "fractional literal on an int column is not pushed");
    }

    @Test
    void doubleColumnAcceptsLongSizedLiteral() {
        // parser-ng-parquet's asDouble rejects Long; the converter must promote it.
        var b = PredicateConverter.bindForTest(ColumnKind.FLOAT64, CompOp.GT, "3000000000");
        assertEquals(Double.class, b.value().getClass());
        assertNotNull(pruning("f64 > 3000000000"));
    }

    @Test
    void floatColumnGetsFloatLiteral() {
        // Finding 5, confirmed by reading: asFloat needs a Float, so Integer/Double literals were dropped.
        var exact = PredicateConverter.bindForTest(ColumnKind.FLOAT32, CompOp.GT, "5");
        assertEquals(Float.class, exact.value().getClass());
        assertEquals(CompOp.GT, exact.op());
        assertNotNull(pruning("f32 > 5"));
    }

    @Test
    void floatColumnRoundsInexactLiteralsOutward() {
        var gt = PredicateConverter.bindForTest(ColumnKind.FLOAT32, CompOp.GT, "0.1");
        assertEquals(CompOp.GE, gt.op());
        assertTrue(((Float) gt.value()).doubleValue() <= 0.1, "lower bound must not exceed the literal");
        var lt = PredicateConverter.bindForTest(ColumnKind.FLOAT32, CompOp.LT, "0.1");
        assertEquals(CompOp.LE, lt.op());
        assertTrue(((Float) lt.value()).doubleValue() >= 0.1, "upper bound must not undercut the literal");
        assertNull(PredicateConverter.bindForTest(ColumnKind.FLOAT32, CompOp.EQ, "0.1"), "no float equals 0.1");
        assertNull(PredicateConverter.bindForTest(ColumnKind.FLOAT32, CompOp.NEQ, "0.1"));
    }

    @Test
    void negativeLiteralAsRenderedByTheParserIsUnderstood() {
        // The parser re-renders unary minus as "- 5"; the old regex never matched it.
        assertNotNull(pruning("i32 > -5"));
        assertNotNull(pruning("i64 BETWEEN -10 AND 10"));
    }

    @Test
    void int64BeyondTwoToTheFiftyThreeUsesTheDoublesValue() {
        var b = PredicateConverter.bindForTest(ColumnKind.INT64, CompOp.GE, "9007199254740993");
        assertEquals(9007199254740992L, b.value());
    }

    @Test
    void stringLiteralsUnescapeDoubledQuotes() {
        var b = PredicateConverter.bindForTest(ColumnKind.STRING, CompOp.EQ, "'it''s'");
        assertEquals("it's", b.value());
    }

    @Test
    void unsupportedKindsAreNeverPushed() {
        assertNull(pruning("d > 1"));
        assertNull(pruning("ts > 1"));
        assertNull(pruning("u > 1"));
    }

    @Test
    void orIsPushedOnlyIfEveryBranchIs() {
        assertNotNull(pruning("i32 > 1 OR i64 < 5"));
        assertNull(pruning("i32 > 1 OR d < 5"));
    }

    @Test
    void andDropsWhatItDoesNotUnderstand() {
        assertNotNull(pruning("i32 > 1 AND d < 5"));
        assertNull(pruning("d < 5 AND ts > 1"));
    }

    @Test
    void columnVersusColumnAndExpressionsAreNotPushed() {
        assertNull(pruning("i32 > i64"));
        assertNull(pruning("i32 * 2 > 10"));
    }

    @Test
    void notInWithUnpushableMemberStillPrunesWithTheRest() {
        assertNotNull(pruning("i32 NOT IN (1, 2.5)"));
        assertNull(pruning("i32 IN (1, 2.5)"));
    }

    // ---------------------------------------------------------------- DECIMAL pruning

    private static final TableSchema DEC_SCHEMA = TableSchema.ofKinds(
            Map.of("d2", ColumnKind.DECIMAL, "wide", ColumnKind.DECIMAL, "i32", ColumnKind.INT32),
            Map.of("d2", new int[] {9, 2}, "wide", new int[] {30, 4}));

    private static Predicate decPruning(String where) {
        return PredicateConverter.convert(SqlParser.parse("SELECT i32 FROM t WHERE " + where).where(), DEC_SCHEMA);
    }

    private static Predicate.Cmp cmpOf(Predicate p) {
        return assertInstanceOf(Predicate.Cmp.class, p);
    }

    @Test
    void decimalLiteralIsPushedAsBigDecimalAtTheColumnScale() {
        Predicate.Cmp c = cmpOf(decPruning("d2 > 5"));
        assertEquals(Predicate.Op.GT, c.op());
        assertEquals(new java.math.BigDecimal("5.00"), c.value());
        assertEquals(new java.math.BigDecimal("5.25"), cmpOf(decPruning("d2 = 5.25")).value());
        assertEquals(new java.math.BigDecimal("-5.00"), cmpOf(decPruning("d2 < - 5")).value());
    }

    @Test
    void decimalLiteralFinerThanTheScaleIsRoundedOutwardForRanges() {
        Predicate.Cmp gt = cmpOf(decPruning("d2 > 1.239"));
        assertEquals(Predicate.Op.GE, gt.op());
        assertEquals(new java.math.BigDecimal("1.23"), gt.value());
        Predicate.Cmp lt = cmpOf(decPruning("d2 < 1.231"));
        assertEquals(Predicate.Op.LE, lt.op());
        assertEquals(new java.math.BigDecimal("1.24"), lt.value());
        Predicate.Cmp neg = cmpOf(decPruning("d2 > - 1.239"));
        assertEquals(new java.math.BigDecimal("-1.24"), neg.value());
    }

    @Test
    void decimalEqualityFinerThanTheScaleAndWideColumnsAreNotPushed() {
        assertNull(decPruning("d2 = 1.239"), "no stored value can equal it");
        assertNull(decPruning("d2 <> 1.239"));
        assertNull(decPruning("wide > 5"), "precision 30 exceeds what a double compares exactly");
        assertNull(decPruning("d2 > 1234567890123456"), "literal beyond 15 digits");
        assertNull(PredicateConverter.convertExact(SqlParser.parse("SELECT i32 FROM t WHERE d2 > 5").where(), DEC_SCHEMA),
                "exactFilter must not rely on a double-compared decimal");
    }

    @Test
    void decimalInAndBetweenPush() {
        assertNotNull(decPruning("d2 BETWEEN 1 AND 2"));
        assertNotNull(decPruning("d2 IN (1, 2.5)"));
        assertNull(decPruning("d2 IN (1, 2.555)"), "one member that cannot be pushed drops the OR");
    }
}
