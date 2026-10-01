package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The WHERE rewriter: which leaves stay with the numeric engine, which become native mask kernels,
 * and which are rejected with a named error. (Kernel evaluation needs Arrow; see ParquetSqlTest.)
 */
class NativeLeavesRewriteTest {

    private static final TableSchema SCHEMA = TableSchema.ofKinds(Map.of(
            "id", ColumnKind.INT64, "x", ColumnKind.FLOAT64, "name", ColumnKind.STRING,
            "flag", ColumnKind.BOOL, "day", ColumnKind.DATE, "blob", ColumnKind.BINARY,
            "other", ColumnKind.STRING));

    private static String m(int i) {
        return NativeLeaves.maskName(i);
    }

    private static NativeLeaves.Result rw(String cond) {
        BoolExpr w = SqlParser.parse("SELECT id FROM t WHERE " + cond).where();
        return NativeLeaves.rewrite(w, SCHEMA);
    }

    private static String engineText(String cond) {
        return BoolExprs.renderFused(rw(cond).where());
    }

    private static String err(String cond) {
        return assertThrows(ParquetSqlException.class, () -> rw(cond)).getMessage();
    }

    @Test
    void numericLeavesStayWithTheEngine() {
        var r = rw("x > 5 AND id < 3");
        assertEquals(0, r.masks().size());
        assertEquals("((x > 5) && (id < 3))", BoolExprs.renderFused(r.where()));
    }

    @Test
    void stringEqualityBecomesAMask() {
        var r = rw("name = 'abc'");
        assertEquals(1, r.masks().size());
        assertEquals("(" + m(0) + " == 1)", BoolExprs.renderFused(r.where()));
    }

    @Test
    void mixedTreeKeepsItsShape() {
        // Under an OR a simple numeric leaf is a two-valued mask too (null is false), see numericLeavesUnderOrAreTwoValued.
        assertEquals("((" + m(0) + " == 1) || (" + m(1) + " == 1))", engineText("x > 5 OR name = 'a'"));
        assertEquals("(((x > 5) && (" + m(0) + " == 1)) && (" + m(1) + " == 1))", engineText("x > 5 AND name = 'a' AND flag"));
    }

    @Test
    void literalOnTheLeftIsFlipped() {
        var r = rw("'m' < name");
        assertEquals(1, r.masks().size());
        assertTrue(r.masks().get(0).description.contains(">"), r.masks().get(0).description);
    }

    @Test
    void betweenOverStringsExpandsToTwoMasks() {
        assertEquals(2, rw("name BETWEEN 'a' AND 'f'").masks().size());
        assertEquals("((" + m(0) + " == 1) || (" + m(1) + " == 1))", engineText("name NOT BETWEEN 'a' AND 'f'"));
    }

    @Test
    void inListIsOneMask() {
        assertEquals(1, rw("name IN ('a', 'b', 'c')").masks().size());
        assertEquals(1, rw("day IN (DATE '2024-01-01', '2024-01-02')").masks().size());
        assertEquals(1, rw("flag IN (TRUE)").masks().size());
    }

    @Test
    void notInWithNullNeverMatches() {
        var r = rw("name NOT IN ('a', NULL)");
        assertTrue(r.masks().get(0).description.contains("never true"), r.masks().get(0).description);
        // A NULL in a positive IN list is simply ignored.
        assertEquals(1, rw("name IN ('a', NULL)").masks().size());
    }

    @Test
    void comparisonWithNullIsAlwaysFalse() {
        assertTrue(rw("x = NULL").masks().get(0).description.contains("NULL"));
        assertTrue(rw("name <> NULL").masks().get(0).description.contains("NULL"));
    }

    @Test
    void likeIlikeRegexp() {
        assertEquals(1, rw("name LIKE 'a%'").masks().size());
        assertTrue(rw("name ILIKE 'A%'").masks().get(0).description.contains("ILIKE"));
        assertTrue(rw("name NOT LIKE 'a%'").masks().get(0).description.contains("NOT"));
        assertTrue(rw("name LIKE 'a!%' ESCAPE '!'").masks().get(0).description.contains("a!%"));
        assertTrue(rw("name REGEXP '^a'").masks().get(0).description.contains("allocates"));
    }

    @Test
    void booleanForms() {
        assertEquals(1, rw("flag").masks().size());
        assertEquals(1, rw("NOT flag").masks().size());
        assertEquals(1, rw("flag = TRUE").masks().size());
        assertEquals(1, rw("flag <> FALSE").masks().size());
        assertTrue(rw("flag IS NOT TRUE").masks().get(0).description.contains("IS NOT TRUE"));
        assertTrue(rw("flag IS FALSE").masks().get(0).description.contains("IS FALSE"));
    }

    @Test
    void dateForms() {
        assertEquals(1, rw("day = DATE '2024-01-01'").masks().size());
        assertEquals(1, rw("day >= '2024-01-01'").masks().size(), "a string literal is read as a date");
        assertEquals(2, rw("day BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'").masks().size());
    }

    @Test
    void isNullIsNativeOnAnyColumnButNotOnExpressions() {
        assertEquals(1, rw("x IS NULL").masks().size());
        assertEquals(1, rw("name IS NOT NULL").masks().size());
        assertEquals(0, rw("x * 2 IS NULL").masks().size(), "expression target is left to the engine");
    }

    @Test
    void errorsNameTheColumnAndType() {
        assertTrue(err("name = 5").contains("'name'") && err("name = 5").contains("STRING"));
        assertTrue(err("id = 'abc'").contains("'id'"));
        assertTrue(err("id = TRUE").contains("'id'"));
        assertTrue(err("day = 'not a date'").contains("DATE"));
        assertTrue(err("flag < TRUE").contains("'flag'"));
        assertTrue(err("flag = 1").contains("'flag'"));
        assertTrue(err("blob = 'x'").contains("BINARY"));
        assertTrue(err("name = other").contains("two columns"));
        assertTrue(err("id LIKE 'a'").contains("STRING"));
        assertTrue(err("name LIKE id").contains("string literal"));
        assertTrue(err("name LIKE 'a' ESCAPE 'ab'").contains("single character"));
        assertTrue(err("name REGEXP '('").contains("Invalid REGEXP"));
        assertTrue(err("x IS TRUE").contains("BOOLEAN"));
        assertTrue(err("name LIKE 'a!'  ESCAPE '!'").contains("escape"));
    }

    @Test
    void maskNamesAreUniqueAndSequential() {
        var r = rw("name = 'a' OR name = 'b' OR flag");
        assertEquals("" + m(0) + "", r.masks().get(0).name);
        assertEquals("" + m(1) + "", r.masks().get(1).name);
        assertEquals("" + m(2) + "", r.masks().get(2).name);
    }

    @Test
    void numericLeavesUnderOrAreTwoValued() {
        var r = rw("x > 5 OR id < 3");
        assertEquals(2, r.masks().size());
        assertEquals("x", r.masks().get(0).widenedColumn());
        assertEquals("id", r.masks().get(1).widenedColumn());
        assertEquals("((" + m(0) + " == 1) || (" + m(1) + " == 1))", BoolExprs.renderFused(r.where()));
        // literal on the left is flipped
        assertEquals("x", rw("5 < x OR id = 1").masks().get(0).widenedColumn());
        // BETWEEN and IN under an OR
        assertEquals(3, rw("x BETWEEN 1 AND 2 OR id = 9").masks().size());
        assertEquals(3, rw("id IN (1, 2) OR x > 0").masks().size());
        assertEquals(1, rw("id NOT IN (1, NULL) OR x > 0").masks().stream().filter(k -> k.widenedColumn() == null).count());
    }

    @Test
    void numericLeavesWithoutAnOrStayFusedInTheEngine() {
        assertEquals(0, rw("x > 5 AND id < 3").masks().size());
        assertEquals(0, rw("x BETWEEN 1 AND 2 AND id IN (1, 2)").masks().size());
    }

    @Test
    void numericExpressionLeavesUnderOrStillGoToTheEngine() {
        var r = rw("x + 1 > 5 OR id < 3");
        assertEquals(1, r.masks().size());
        assertEquals("id", r.masks().get(0).widenedColumn());
    }
}
