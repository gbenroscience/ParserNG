package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Delimited identifiers ({@code "my col"}, {@code `my col`}), the {@code ||} operator, and the new meaning of double quotes. */
class QuotedIdentifierGrammarTest {

    private static final String REF = QuotedIdentifiers.encode("my col");

    @Test
    void referencesRoundTripAndRejectAnythingElse() {
        assertTrue(REF.matches("pngq[a-p]+z"));
        assertEquals("my col", QuotedIdentifiers.decode(REF));
        assertEquals("naïve 列", QuotedIdentifiers.decode(QuotedIdentifiers.encode("naïve 列")));
        assertNull(QuotedIdentifiers.decode("id"));
        assertNull(QuotedIdentifiers.decode("pngqab"));
        assertNull(QuotedIdentifiers.decode("pngqaz"));
        assertEquals("id", QuotedIdentifiers.real("id"));
        assertEquals("my col + 1", QuotedIdentifiers.decodeAll(REF + " + 1"));
    }

    @Test
    void plainNamesStayRawAndOthersBecomeReferences() {
        assertEquals("id", QuotedIdentifiers.reference("id"));
        assertNotEquals("select", QuotedIdentifiers.reference("select"));
        assertNotEquals("1st", QuotedIdentifiers.reference("1st"));
        assertNotEquals("a-b", QuotedIdentifiers.reference("a-b"));
    }

    @Test
    void quotedIdentifiersParseEverywhereAnExpressionDoes() {
        SelectStatement s = SqlParser.parse(
                "SELECT \"my col\", `a-b` AS \"out name\", \"my col\" + 1 FROM t WHERE \"my col\" > 3 ORDER BY \"my col\"");
        assertEquals(REF, s.items().get(0).exprText());
        assertEquals("my col", s.items().get(0).outputName(), "the output keeps the real name");
        assertEquals("out name", s.items().get(1).outputName());
        assertEquals("my col + 1", s.items().get(2).outputName());
        assertEquals(REF, s.orderBy().get(0).exprText());
        assertTrue(s.where().toString().contains(REF));
        assertEquals("SUM(my col)", SqlParser.parse("SELECT SUM(\"my col\") FROM t").items().get(0).outputName());
    }

    @Test
    void aQuotedPlainNameIsJustThatName() {
        SelectStatement s = SqlParser.parse("SELECT \"id\" FROM t");
        assertEquals("id", s.items().get(0).exprText());
        assertEquals("id", s.items().get(0).outputName());
    }

    @Test
    void theDelimiterIsEscapedByDoublingIt() {
        SelectStatement s = SqlParser.parse("SELECT \"a\"\"b\", `c``d` FROM t");
        assertEquals("a\"b", QuotedIdentifiers.decode(s.items().get(0).exprText()));
        assertEquals("c`d", QuotedIdentifiers.decode(s.items().get(1).exprText()));
    }

    @Test
    void doubleQuotesNoLongerMeanStringLiterals() {
        // a string literal is single-quoted only; a double-quoted name is an identifier
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE name = 'a\"b'");
        assertTrue(s.where().toString().contains("a\"b"));
        s = SqlParser.parse("SELECT x FROM t WHERE name = \"other col\"");
        assertTrue(s.where().toString().contains(QuotedIdentifiers.encode("other col")));
    }

    @Test
    void aQuotedNameAfterFromIsATableName() {
        SelectStatement s = SqlParser.parse("SELECT x FROM \"t1\"");
        assertEquals("t1", s.table());
        assertEquals(SelectStatement.TableKind.NAME, s.tableKind());
    }

    @Test
    void malformedIdentifiersAreRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT \"\" FROM t"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT \"abc FROM t"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT `abc FROM t"));
    }

    @Test
    void concatenationOperatorParsesAndASinglePipeDoesNot() {
        assertEquals("a || b || 'x'", SqlParser.parse("SELECT a || b || 'x' AS c FROM t").items().get(0).exprText());
        assertTrue(SqlParser.parse("SELECT a FROM t WHERE a || b = 'xy'").where().toString().contains("a || b"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT a | b FROM t"));
    }

    @Test
    void stringFunctionsParseAsOrdinaryFunctionCalls() {
        SelectStatement s = SqlParser.parse("SELECT LOWER(name), SUBSTRING(name, 2, 3), LENGTH(TRIM(name)), CONCAT(a, '-', b) FROM t");
        assertEquals("SUBSTRING(name, 2, 3)", s.items().get(1).exprText());
        assertEquals("LENGTH(TRIM(name))", s.items().get(2).exprText());
        assertEquals("CONCAT(a, '-', b)", s.items().get(3).exprText());
    }
}
