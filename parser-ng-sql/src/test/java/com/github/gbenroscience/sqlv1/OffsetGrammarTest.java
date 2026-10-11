package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Grammar for {@code LIMIT n OFFSET m}, {@code OFFSET m [LIMIT n]}; OFFSET stays usable as a column name. */
class OffsetGrammarTest {

    @Test
    void limitThenOffset() {
        SelectStatement s = SqlParser.parse("SELECT a FROM t ORDER BY a LIMIT 5 OFFSET 2");
        assertEquals(5, s.limit());
        assertEquals(2, s.offset());
        assertEquals(1, s.orderBy().size());
        assertEquals("a", s.orderBy().get(0).exprText(), "the order item must not swallow OFFSET");
    }

    @Test
    void offsetAloneAndOffsetBeforeLimit() {
        SelectStatement s = SqlParser.parse("SELECT a FROM t OFFSET 7");
        assertNull(s.limit());
        assertEquals(7, s.offset());
        s = SqlParser.parse("SELECT a FROM t ORDER BY a OFFSET 2 LIMIT 3");
        assertEquals(3, s.limit());
        assertEquals(2, s.offset());
        assertEquals("a", s.orderBy().get(0).exprText());
        s = SqlParser.parse("SELECT a FROM t WHERE a > 1 OFFSET 2");
        assertEquals("1", ((com.github.gbenroscience.sqlv1.ast.ComparisonExpr) s.where()).right());
        assertEquals(2, s.offset());
    }

    @Test
    void offsetRowsKeywordAndDefault() {
        assertEquals(4, SqlParser.parse("SELECT a FROM t OFFSET 4 ROWS").offset());
        assertEquals(1, SqlParser.parse("SELECT a FROM t LIMIT 2 OFFSET 1 ROW").offset());
        assertEquals(0, SqlParser.parse("SELECT a FROM t LIMIT 2").offset());
    }

    @Test
    void aColumnNamedOffsetIsStillAColumn() {
        SelectStatement s = SqlParser.parse("SELECT a, offset FROM t WHERE offset > 3 ORDER BY offset");
        assertEquals(0, s.offset());
        assertEquals("offset", s.items().get(1).exprText());
        assertEquals("offset", s.orderBy().get(0).exprText());
    }

    @Test
    void malformedOffsetIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT a FROM t LIMIT 5 OFFSET"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT a FROM t OFFSET 1.5"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT a FROM t OFFSET 99999999999"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT a FROM t OFFSET 1 OFFSET 2"));
    }

    @Test
    void coalesceAndNullifParseAsFunctionCalls() {
        SelectStatement s = SqlParser.parse("SELECT COALESCE(a, 0) + 1 AS x, NULLIF(b, -1) FROM t WHERE COALESCE(a, b) > 5");
        assertEquals("COALESCE(a, 0) + 1", s.items().get(0).exprText());
        assertEquals("NULLIF(b, - 1)", s.items().get(1).exprText());
    }
}
