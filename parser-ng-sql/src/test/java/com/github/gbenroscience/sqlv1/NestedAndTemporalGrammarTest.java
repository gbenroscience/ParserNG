package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Nested field paths, EXTRACT, the new CAST targets and INTERVAL text. */
class NestedAndTemporalGrammarTest {

    private static String path(String ref) {
        String real = QuotedIdentifiers.decode(ref);
        return real == null ? ref : real;
    }

    @Test
    void nestedPathsBecomeOneReferenceToTheWholePath() {
        SelectStatement s = SqlParser.parse("SELECT s.field, items[0], m['key'], s.items[2].price, \"my s\".\"f g\" FROM t WHERE s.a > 3 ORDER BY s.a");
        assertEquals("s.field", path(s.items().get(0).exprText()));
        assertEquals("s.field", s.items().get(0).outputName(), "the default output name is the path");
        assertEquals("items[0]", path(s.items().get(1).exprText()));
        assertEquals("m['key']", path(s.items().get(2).exprText()));
        assertEquals("s.items[2].price", path(s.items().get(3).exprText()));
        assertEquals("my s.f g", path(s.items().get(4).exprText()));
        assertEquals("s.a", path(s.orderBy().get(0).exprText()));
        assertTrue(s.where().toString().contains(QuotedIdentifiers.encode("s.a")));
    }

    @Test
    void aMapKeyWithAQuoteKeepsItDoubled() {
        assertEquals("m['it''s']", path(SqlParser.parse("SELECT m['it''s'] FROM t").items().get(0).exprText()));
    }

    @Test
    void cardinalityIsAnOrdinaryCallAndAggregatesTakePaths() {
        SelectStatement s = SqlParser.parse("SELECT CARDINALITY(s.items), SUM(s.v) FROM t");
        assertTrue(s.items().get(0).exprText().startsWith("CARDINALITY("));
        assertEquals(QuotedIdentifiers.encode("s.v"), s.items().get(1).aggregate().argExprText());
    }

    @Test
    void malformedPathsAreRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT items[1.5] FROM t"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT items[ FROM t"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT s. FROM t"));
    }

    @Test
    void extractKeepsSqlText() {
        SelectStatement s = SqlParser.parse("SELECT EXTRACT(year FROM ts), EXTRACT(MONTH FROM CAST(x AS DATE)) AS m FROM t");
        assertEquals("EXTRACT(YEAR FROM ts)", s.items().get(0).exprText());
        assertEquals("EXTRACT(MONTH FROM CAST(x AS DATE))", s.items().get(1).exprText());
        assertEquals("m", s.items().get(1).outputName());
    }

    @Test
    void castKeepsItsSqlFormForTheTypesTheBridgeEvaluates() {
        SelectStatement s = SqlParser.parse(
                "SELECT CAST(x AS DATE), CAST(x AS timestamp), CAST(x AS DECIMAL(10, 2)), CAST(x AS VARCHAR(20)) FROM t");
        assertEquals("CAST(x AS DATE)", s.items().get(0).exprText());
        assertEquals("CAST(x AS TIMESTAMP)", s.items().get(1).exprText());
        assertEquals("CAST(x AS DECIMAL(10,2))", s.items().get(2).exprText());
        assertEquals("CAST(x AS VARCHAR)", s.items().get(3).exprText());
    }

    @Test
    void numericCastsAreStillDesugared() {
        SelectStatement s = SqlParser.parse("SELECT CAST(x AS DECIMAL), CAST(x AS INT), CAST(x AS DOUBLE) FROM t");
        assertEquals("(x)", s.items().get(0).exprText());
        assertEquals("((x) - ((x) % 1))", s.items().get(1).exprText());
        assertEquals("(x)", s.items().get(2).exprText());
    }

    @Test
    void intervalStaysAsTextInsideTheExpression() {
        SelectStatement s = SqlParser.parse(
                "SELECT a FROM t WHERE ts > TIMESTAMP '2024-01-01 00:00:00' - INTERVAL '7 days' AND d + INTERVAL '1' DAY < DATE '2024-02-01'");
        assertTrue(s.where().toString().contains("INTERVAL '7 days'"));
        assertTrue(s.where().toString().contains("INTERVAL '1' DAY"));
    }

    @Test
    void aQualifiedNameIsNowAPath() {
        assertEquals("t.x", path(SqlParser.parse("SELECT t.x FROM t").items().get(0).exprText()));
    }
}
