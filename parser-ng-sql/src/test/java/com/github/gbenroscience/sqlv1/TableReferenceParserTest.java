package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import com.github.gbenroscience.sqlv1.ast.SelectStatement.TableKind;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** The {@code FROM} forms added for the Parquet bridge. A bare name keeps its old meaning. */
class TableReferenceParserTest {

    @Test
    void bareNameIsUnchanged() {
        SelectStatement st = SqlParser.parse("SELECT x FROM points");
        assertEquals(TableKind.NAME, st.tableKind());
        assertEquals("points", st.table());
        assertEquals(List.of(), st.tablePaths());
    }

    @Test
    void quotedPathIsAFileReference() {
        SelectStatement st = SqlParser.parse("SELECT x FROM 'data/a.parquet' WHERE x > 1");
        assertEquals(TableKind.FILE, st.tableKind());
        assertEquals("data/a.parquet", st.table());
        assertEquals(List.of("data/a.parquet"), st.tablePaths());
    }

    @Test
    void readParquetTakesOneOrMorePaths() {
        SelectStatement st = SqlParser.parse("SELECT COUNT(*) FROM read_parquet('a/*.parquet', 'b.parquet')");
        assertEquals(TableKind.READ_PARQUET, st.tableKind());
        assertEquals(List.of("a/*.parquet", "b.parquet"), st.tablePaths());
        assertEquals("a/*.parquet", st.table());
    }

    @Test
    void readParquetNameIsCaseInsensitive() {
        assertEquals(TableKind.READ_PARQUET, SqlParser.parse("SELECT x FROM READ_PARQUET('a')").tableKind());
    }

    @Test
    void unknownTableFunctionIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM read_csv('a')"));
    }

    @Test
    void readParquetNeedsQuotedArgumentsAndClosingParen() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM read_parquet(a)"));
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM read_parquet('a'"));
    }

    @Test
    void legacyEightArgumentConstructorStillWorks() {
        SelectStatement st = SqlParser.parse("SELECT x FROM t");
        SelectStatement copy = new SelectStatement(st.selectAll(), st.items(), st.table(), st.where(),
                st.groupBy(), st.having(), st.orderBy(), st.limit());
        assertEquals(TableKind.NAME, copy.tableKind());
    }
}
