package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Grammar added for string and boolean predicates: LIKE, ILIKE, REGEXP, IS [NOT] TRUE/FALSE, bare boolean. */
class PredicateGrammarTest {

    private static BoolExpr where(String cond) {
        return SqlParser.parse("SELECT x FROM t WHERE " + cond).where();
    }

    @Test
    void like() {
        assertEquals(new LikeExpr("name", "'a%'", null, false, false), where("name LIKE 'a%'"));
        assertEquals(new LikeExpr("name", "'a%'", null, false, true), where("name NOT LIKE 'a%'"));
    }

    @Test
    void ilikeAndEscape() {
        assertEquals(new LikeExpr("name", "'a!%'", "'!'", true, false), where("name ILIKE 'a!%' ESCAPE '!'"));
    }

    @Test
    void regexpAndRlike() {
        assertEquals(new RegexExpr("name", "'^a.*z$'", false), where("name REGEXP '^a.*z$'"));
        assertEquals(new RegexExpr("name", "'x'", true), where("name NOT RLIKE 'x'"));
    }

    @Test
    void isTrueIsFalse() {
        assertEquals(new TruthExpr("flag", true, false), where("flag IS TRUE"));
        assertEquals(new TruthExpr("flag", true, true), where("flag IS NOT TRUE"));
        assertEquals(new TruthExpr("flag", false, false), where("flag IS FALSE"));
        assertEquals(new TruthExpr("flag", false, true), where("flag IS NOT FALSE"));
    }

    @Test
    void isNullStillWorks() {
        assertEquals(new IsNullExpr("flag", true), where("flag IS NOT NULL"));
    }

    @Test
    void bareBooleanIsEqualsTrue() {
        assertEquals(new ComparisonExpr("flag", CompOp.EQ, "TRUE"), where("flag"));
        assertEquals(new AndExpr(new ComparisonExpr("flag", CompOp.EQ, "TRUE"), new ComparisonExpr("x", CompOp.GT, "1")),
                where("flag AND x > 1"));
        assertEquals(new OrExpr(new ComparisonExpr("x", CompOp.GT, "1"), new ComparisonExpr("flag", CompOp.EQ, "TRUE")),
                where("x > 1 OR (flag)"));
    }

    @Test
    void notBareBooleanKeepsThreeValuedSemantics() {
        // NOT flag must become flag <> TRUE (null stays unknown), not "IS NOT TRUE" (which would keep nulls).
        assertEquals(new ComparisonExpr("flag", CompOp.NEQ, "TRUE"), where("NOT flag"));
    }

    @Test
    void negatingTheTwoValuedIsFormsFlipsTheNegation() {
        assertEquals(new TruthExpr("flag", true, true), where("NOT (flag IS TRUE)"));
        assertEquals(new LikeExpr("n", "'a'", null, false, true), where("NOT (n LIKE 'a')"));
        assertEquals(new RegexExpr("n", "'a'", true), where("NOT (n REGEXP 'a')"));
    }

    @Test
    void bareIdentifierStillFailsInsideExpressionsAndSelectItems() {
        // The boolean-first trial in select items / function arguments must keep failing on a plain x.
        assertEquals("x", SqlParser.parse("SELECT x FROM t").items().get(0).exprText());
        assertEquals("sin(x)", SqlParser.parse("SELECT sin(x) FROM t").items().get(0).exprText());
    }

    @Test
    void bareNonIdentifierExpressionIsStillAnError() {
        assertThrows(SqlSyntaxException.class, () -> where("x + 1"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM t WHERE sin(x)"));
    }

    @Test
    void malformedTails() {
        assertThrows(SqlSyntaxException.class, () -> where("name LIKE"));
        assertThrows(SqlSyntaxException.class, () -> where("name NOT"));
        assertThrows(SqlSyntaxException.class, () -> where("flag IS 5"));
    }

    @Test
    void combinesWithGroupingAndOrdering() {
        SelectStatement st = SqlParser.parse("SELECT x FROM 'f.parquet' WHERE name LIKE 'a%' AND flag ORDER BY x LIMIT 5");
        assertTrue(st.where() instanceof AndExpr);
        assertEquals(Integer.valueOf(5), st.limit());
    }

    @Test
    void newKeywordsAreCaseInsensitive() {
        assertEquals(new LikeExpr("n", "'a'", null, false, false), where("n like 'a'"));
        assertEquals(new RegexExpr("n", "'a'", false), where("n regexp 'a'"));
    }

    @Test
    void likeIsNotRenderableByTheNumericEngine() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> BoolExprs.renderLeaf(where("n LIKE 'a'")));
        assertTrue(e.getMessage().contains("ParquetSql"));
    }
}
