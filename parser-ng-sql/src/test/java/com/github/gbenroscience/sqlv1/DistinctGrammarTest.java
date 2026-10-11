package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Grammar for {@code SELECT DISTINCT} and {@code COUNT(DISTINCT expr)}; DISTINCT stays usable as a column name. */
class DistinctGrammarTest {

    @Test
    void selectDistinctSetsTheFlag() {
        SelectStatement s = SqlParser.parse("SELECT DISTINCT a, b FROM t");
        assertTrue(s.distinct());
        assertEquals(2, s.items().size());
        assertTrue(SqlParser.parse("select distinct * from t").selectAll());
        assertFalse(SqlParser.parse("SELECT a FROM t").distinct());
    }

    @Test
    void aColumnNamedDistinctIsStillAColumn() {
        SelectStatement s = SqlParser.parse("SELECT distinct FROM t");
        assertFalse(s.distinct());
        assertEquals("distinct", s.items().get(0).exprText());
        assertFalse(SqlParser.parse("SELECT distinct, b FROM t").distinct());
        assertEquals("d", SqlParser.parse("SELECT distinct AS d FROM t").items().get(0).alias());
        assertFalse(SqlParser.parse("SELECT COUNT(distinct) FROM t").items().get(0).aggregate().distinct());
    }

    @Test
    void countDistinctIsParsedAndNamed() {
        AggregateSpec a = SqlParser.parse("SELECT COUNT(DISTINCT a) FROM t").items().get(0).aggregate();
        assertTrue(a.distinct());
        assertEquals(AggFunc.COUNT, a.func());
        assertEquals("a", a.argExprText());
        assertEquals("COUNT(DISTINCT a)", a.defaultOutputName());
        assertFalse(SqlParser.parse("SELECT COUNT(*), MIN(name) FROM t").items().get(0).aggregate().distinct());
    }

    @Test
    void distinctOnOtherAggregatesAndStarIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT SUM(DISTINCT a) FROM t"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT COUNT(DISTINCT *) FROM t"));
    }

    @Test
    void distinctComposesWithWhereOrderAndLimit() {
        SelectStatement s = SqlParser.parse("SELECT DISTINCT a FROM t WHERE a > 1 ORDER BY a LIMIT 3");
        assertTrue(s.distinct());
        assertEquals(3, s.limit());
        assertEquals(1, s.orderBy().size());
    }
}
