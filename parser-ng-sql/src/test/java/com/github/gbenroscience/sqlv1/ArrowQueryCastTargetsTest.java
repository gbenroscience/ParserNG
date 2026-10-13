package com.github.gbenroscience.sqlv1;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link SqlParser#parse(String)} lets the Parquet-only CAST targets through (the bridge evaluates them), so
 * {@link ArrowQuery#compile(String)} must reject them itself with a clear message instead of failing later inside ParserNG.
 */
class ArrowQueryCastTargetsTest {

    @Test
    void parquetOnlyCastTargetsAreRejectedByArrowQuery() {
        for (String type : new String[] {"VARCHAR", "varchar(20)", "CHAR", "TEXT", "STRING", "DATE", "TIMESTAMP", "DECIMAL(10,2)", "NUMERIC(8, 3)"}) {
            String sql = "SELECT CAST(x AS " + type + ") AS v FROM t";
            SqlSyntaxException e = assertThrows(SqlSyntaxException.class, () -> ArrowQuery.compile(sql), sql);
            assertTrue(e.getMessage().contains("only supported by ParquetSql"), e.getMessage());
        }
    }

    @Test
    void aNestedOrLaterParquetOnlyCastIsStillFound() {
        assertThrows(SqlSyntaxException.class,
                () -> ArrowQuery.compile("SELECT CAST(CAST(x AS INT) AS VARCHAR) AS v FROM t"));
        assertThrows(SqlSyntaxException.class,
                () -> ArrowQuery.compile("SELECT x FROM t WHERE CAST(y AS DATE) > 1"));
    }

    @Test
    void numericCastsStillCompile() {
        for (String type : new String[] {"INT", "INTEGER", "BIGINT", "SMALLINT", "LONG", "DOUBLE", "FLOAT", "REAL", "NUMERIC", "DECIMAL"}) {
            assertDoesNotThrow(() -> ArrowQuery.compile("SELECT CAST(x AS " + type + ") AS v FROM t").close(), type);
        }
    }

    @Test
    void aStringLiteralThatLooksLikeACastIsNotRejected() {
        assertDoesNotThrow(() -> SqlParser.requireNumericCastTargets("SELECT x FROM t WHERE name = 'CAST(x AS VARCHAR)'"));
    }
}
