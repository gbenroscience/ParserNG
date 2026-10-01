package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Pure planning tests: no Arrow, no file. */
class ScanPlannerTest {

    private static final List<String> FILE = List.of("id", "x", "y", "name");
    private static final TableSchema SCHEMA = TableSchema.ofKinds(Map.of(
            "id", ColumnKind.INT64, "x", ColumnKind.FLOAT64, "y", ColumnKind.FLOAT32, "name", ColumnKind.STRING));

    private static ScanPlanner.ScanPlan plan(String sql) {
        return ScanPlanner.plan(SqlParser.parse(sql), FILE, FILE, SCHEMA);
    }

    @Test
    void countStarWithoutWhereIsMetadataOnly() {
        var p = plan("SELECT COUNT(*) FROM t");
        assertTrue(p.metadataOnly());
        assertEquals(List.of(), p.columns(), "no column may be decoded");
        assertNull(p.pruning());
    }

    @Test
    void repeatedCountStarIsStillMetadataOnly() {
        assertTrue(plan("SELECT COUNT(*) AS a, COUNT(*) AS b FROM t").metadataOnly());
    }

    @Test
    void anythingElseIsNotMetadataOnly() {
        assertFalse(plan("SELECT COUNT(*) FROM t WHERE x > 1").metadataOnly());
        assertFalse(plan("SELECT COUNT(x) FROM t").metadataOnly());
        assertFalse(plan("SELECT SUM(x) FROM t").metadataOnly());
        assertFalse(plan("SELECT COUNT(*) FROM t GROUP BY id").metadataOnly());
        assertFalse(plan("SELECT x FROM t").metadataOnly());
        assertFalse(plan("SELECT COUNT(*) FROM t LIMIT 0").metadataOnly());
    }

    @Test
    void onlyReferencedColumnsAreDecodedInFileOrder() {
        var p = plan("SELECT name, x * 2 AS d FROM t WHERE id > 5");
        assertEquals(List.of("id", "x", "name"), p.columns());
    }

    @Test
    void selectStarDecodesEverything() {
        assertNull(plan("SELECT * FROM t").columns());
    }

    @Test
    void floatColumnNowGetsAPruningPredicate() {
        // Finding 5: with the old converter this was null (Integer literal on a float32 column).
        assertNotNull(plan("SELECT y FROM t WHERE y > 5").pruning());
    }

    @Test
    void streamingVersusMaterializing() {
        assertTrue(plan("SELECT x FROM t WHERE x > 1 LIMIT 5").streamable());
        assertEquals(Integer.valueOf(5), plan("SELECT x FROM t LIMIT 5").limit());
        assertFalse(plan("SELECT x FROM t ORDER BY x").streamable());
        assertFalse(plan("SELECT SUM(x) FROM t").streamable());
    }

    @Test
    void columnlessQueryFallsBackToAReadableColumn() {
        assertEquals(List.of("id"), plan("SELECT 1 + 1 AS two FROM t").columns());
    }
}
