package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.ScanMetrics;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end tests of the Parquet bridge over real files. Findings 2 to 5 of the task are
 * reproduced here as ordinary tests: each fails against the first bridge (which fed native
 * vectors straight into a Float8-only engine) and passes now.
 *
 * <p>These tests need Arrow, parser-ng-arrow, parser-ng-parquet and parquet-java on the classpath.
 */
class ParquetSqlTest {

    private static final int ROWS = 20_000;
    private static Path dir;
    private static Path main;

    private static synchronized Path main() throws IOException {
        if (main == null) {
            dir = Files.createTempDirectory("parquet-sql-test");
            main = dir.resolve("main.parquet");
            ParquetFixtures.writeMain(main, ROWS);
        }
        return main;
    }

    private static String str(VectorSchemaRoot r, String col, int i) {
        VarCharVector v = (VarCharVector) r.getVector(col);
        return v.isNull(i) ? null : new String(v.get(i), StandardCharsets.UTF_8);
    }

    // ---------------------------------------------------------------- finding 2: type failure

    @Test
    void int64AndBooleanColumnsWork() throws Exception {
        // The shipped example (int64 id, boolean flag) failed in the first bridge.
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT id, flag FROM t WHERE id < 5", a)) {
            assertEquals(5, r.getRowCount());
            assertTrue(r.getVector("id") instanceof BigIntVector, "int64 stays int64 (passthrough)");
            assertTrue(r.getVector("flag") instanceof BitVector);
            for (int i = 0; i < 5; i++) assertEquals((long) i, ((BigIntVector) r.getVector("id")).get(i));
            assertTrue(r.getVector("flag").isNull(0), "row 0: i%7==0, null preserved");
            assertEquals(0, ((BitVector) r.getVector("flag")).get(1));
            assertEquals(1, ((BitVector) r.getVector("flag")).get(3));
        }
    }

    @Test
    void everyColumnTypeRoundTripsWithNulls() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(),
                     "SELECT id, i32, f32, f64, flag, name, day, ts, dec32, dec64, decf FROM t WHERE id IN (0, 1, 22, 13)", a)) {
            assertEquals(4, r.getRowCount());
            int idx = -1;
            for (int i = 0; i < 4; i++) if (((BigIntVector) r.getVector("id")).get(i) == 22) idx = i;
            assertTrue(idx >= 0);
            assertEquals(22 % 100, ((IntVector) r.getVector("i32")).get(idx));
            assertEquals(11.0f, ((Float4Vector) r.getVector("f32")).get(idx));
            assertEquals(33.0, ((Float8Vector) r.getVector("f64")).get(idx));
            assertTrue(r.getVector("name").isNull(idx), "22 % 11 == 0 -> null name");
            assertEquals(19000 + 22, ((DateDayVector) r.getVector("day")).get(idx));
            assertEquals(ParquetFixtures.TS_BASE + 22_000_000L, ((TimeStampVector) r.getVector("ts")).get(idx));
            assertEquals(new BigDecimal("22.05"), ((DecimalVector) r.getVector("dec32")).getObject(idx));
            assertEquals(new BigDecimal("22.0000"), ((DecimalVector) r.getVector("dec64")).getObject(idx));
            assertEquals(new BigDecimal("0.022"), ((DecimalVector) r.getVector("decf")).getObject(idx));
            for (int i = 0; i < 4; i++) {
                if (((BigIntVector) r.getVector("id")).get(i) == 0) {
                    assertTrue(r.getVector("i32").isNull(i));
                    assertTrue(r.getVector("flag").isNull(i));
                    assertTrue(r.getVector("name").isNull(i));
                }
                if (((BigIntVector) r.getVector("id")).get(i) == 1) assertEquals("n1", str(r, "name", i));
            }
        }
    }

    @Test
    void numericExpressionsAndNullsMatchAJavaReference() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(),
                     "SELECT id, i32 + 1 AS p, f32 * 2 AS d FROM t WHERE f64 < 3000 AND id >= 10", a)) {
            List<Long> expectIds = new ArrayList<>();
            for (long i = 10; i < ROWS; i++) if (i * 1.5 < 3000) expectIds.add(i);
            assertEquals(expectIds.size(), r.getRowCount());
            BigIntVector id = (BigIntVector) r.getVector("id");
            Float8Vector p = (Float8Vector) r.getVector("p");
            Float8Vector d = (Float8Vector) r.getVector("d");
            for (int k = 0; k < r.getRowCount(); k++) {
                long i = expectIds.get(k);
                assertEquals(i, id.get(k));
                if (i % 10 == 0) assertTrue(p.isNull(k), "null i32 propagates through i32 + 1");
                else assertEquals((i % 100) + 1.0, p.get(k), 0.0);
                assertEquals(i * 0.5 * 2, d.get(k), 0.0);
            }
        }
    }

    @Test
    void orWithNullsFollowsSqlThreeValuedLogic() throws Exception {
        // Rows where i32 is null must still match through the other branch.
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT id FROM t WHERE i32 > 1000 OR id < 25", a)) {
            assertEquals(25, r.getRowCount());
        }
    }

    // ---------------------------------------------------------------- finding 5 + pruning

    @Test
    void whereOnSortedColumnSkipsRowGroups() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t WHERE id < 100")) {
            List<Long> got = new ArrayList<>();
            p.stream(a, b -> { for (int i = 0; i < b.getRowCount(); i++) got.add(((BigIntVector) b.getVector("id")).get(i)); });
            assertEquals(100, got.size());
            ScanMetrics m = p.lastMetrics();
            assertTrue(m.rowGroupsInFile() > 1, "fixture must span several row groups: " + m);
            assertTrue(m.rowGroupsSkipped() > 0, "expected pruning: " + m);
            assertTrue(m.rowGroupsRead() < m.rowGroupsInFile());
        }
    }

    @Test
    void floatColumnPrunesToo() throws Exception {
        // Finding 5: a plain integer literal on a float32 column used to disable pruning.
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t WHERE f32 < 50")) {
            p.stream(a, b -> { });
            assertTrue(p.lastMetrics().rowGroupsSkipped() > 0, "expected pruning: " + p.lastMetrics());
        }
    }

    @Test
    void pruningNeverChangesTheAnswer() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT id FROM t WHERE f32 > 0.1 AND f32 < 30.3", a)) {
            int expect = 0;
            for (int i = 0; i < ROWS; i++) if ((double) (i * 0.5f) > 0.1 && (double) (i * 0.5f) < 30.3) expect++;
            assertEquals(expect, r.getRowCount());
        }
    }

    // ---------------------------------------------------------------- modes, limit, cap

    @Test
    void streamingAndMaterializingAgree() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id, f64 * 2 AS d FROM t WHERE id % 3 = 0")) {
            assertTrue(p.scanPlan().streamable());
            long[] rows = {0};
            p.stream(a, b -> rows[0] += b.getRowCount());
            try (VectorSchemaRoot all = p.execute(a)) {
                assertEquals(rows[0], all.getRowCount());
            }
        }
    }

    @Test
    void orderByLimitMaterializesAndSorts() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id, name FROM t ORDER BY id DESC LIMIT 3")) {
            assertFalse(p.scanPlan().streamable());
            assertThrows(UnsupportedOperationException.class, () -> p.stream(a, b -> { }));
            try (VectorSchemaRoot r = p.execute(a)) {
                assertEquals(3, r.getRowCount());
                assertEquals((long) ROWS - 1, ((BigIntVector) r.getVector("id")).get(0));
                assertEquals((long) ROWS - 3, ((BigIntVector) r.getVector("id")).get(2));
                assertEquals("n" + (ROWS - 1), str(r, "name", 0));
            }
        }
    }

    @Test
    void limitStopsReadingEarly() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t LIMIT 5")) {
            long[] rows = {0};
            p.stream(a, b -> rows[0] += b.getRowCount());
            assertEquals(5, rows[0]);
            ScanMetrics m = p.lastMetrics();
            assertTrue(m.rowGroupsRead() < m.rowGroupsAfterPruning(), "should not read every row group: " + m);
        }
    }

    @Test
    void materializationCapFailsLoudly() throws Exception {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withMaxMaterializedRows(10);
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t ORDER BY id", o)) {
            ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> p.execute(a));
            assertTrue(e.getMessage().contains("maxMaterializedRows"));
        }
    }

    @Test
    void aggregatesRunOverWidenedColumns() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT SUM(f64) AS s, COUNT(*) AS c FROM t WHERE id < 100", a)) {
            assertEquals(1, r.getRowCount());
            assertEquals(1.5 * (99 * 100 / 2), ((Float8Vector) r.getVector("s")).get(0), 1e-9);
            assertEquals(100.0, ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
    }

    @Test
    void scanOptionsDoNotChangeResults() throws Exception {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withParallelism(2).withBatchSize(500).withMetrics(true);
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t WHERE id % 2 = 0", o)) {
            long[] rows = {0};
            p.stream(a, b -> rows[0] += b.getRowCount());
            assertEquals(ROWS / 2, rows[0]);
            assertEquals(2, p.lastMetrics().parallelism());
        }
    }

    @Test
    void exactFilterFiltersInTheScanWithSameResult() throws Exception {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withExactFilter(true);
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t WHERE id >= 100 AND id < 200", o);
             VectorSchemaRoot r = p.execute(a)) {
            assertEquals(100, r.getRowCount());
            assertTrue(p.explain().contains("exactFilter"));
        }
    }

    // ---------------------------------------------------------------- footer-only

    @Test
    void countStarIsAnsweredFromTheFooter() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT COUNT(*) AS n FROM t");
             VectorSchemaRoot r = p.execute(a)) {
            assertTrue(p.scanPlan().metadataOnly());
            assertEquals((double) ROWS, ((Float8Vector) r.getVector("n")).get(0), 0.0);
            assertNull(p.lastMetrics(), "no scan ran, so no data page was decoded");
        }
    }

    // ---------------------------------------------------------------- clear errors

    // ---------------------------------------------------------------- strings, booleans, dates, timestamps

    private static List<Long> ids(String where) throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT id FROM t WHERE " + where, a)) {
            List<Long> out = new ArrayList<>();
            BigIntVector id = (BigIntVector) r.getVector("id");
            for (int i = 0; i < r.getRowCount(); i++) out.add(id.get(i));
            return out;
        }
    }

    private static String name(long i) { return i % 11 == 0 ? null : "n" + i; }
    private static Boolean flag(long i) { return i % 7 == 0 ? null : (Boolean) (i % 3 == 0); }

    private interface RowTest { boolean test(long i); }

    private static List<Long> expect(RowTest t) {
        List<Long> out = new ArrayList<>();
        for (long i = 0; i < ROWS; i++) if (t.test(i)) out.add(i);
        return out;
    }

    @Test
    void stringEqualityInAndRange() throws Exception {
        assertEquals(List.of(15L), ids("name = 'n15'"));
        assertEquals(List.of(1L, 2L), ids("name IN ('n1', 'n2', 'n11')"), "n11 does not exist: row 11 is null");
        assertEquals(expect(i -> name(i) != null && !name(i).equals("n5")), ids("name <> 'n5'"), "null is not <> 'n5'");
        assertEquals(expect(i -> name(i) != null && name(i).compareTo("n5") < 0), ids("name < 'n5'"));
        assertEquals(expect(i -> name(i) != null && name(i).compareTo("n2") >= 0 && name(i).compareTo("n3") <= 0),
                ids("name BETWEEN 'n2' AND 'n3'"));
    }

    @Test
    void quotedApostropheAndEmptyString() throws Exception {
        assertEquals(List.of(), ids("name = 'it''s'"));
        assertEquals(List.of(), ids("name = ''"));
    }

    @Test
    void likeIlikeAndRegexpWithNulls() throws Exception {
        assertEquals(expect(i -> name(i) != null && name(i).startsWith("n1")), ids("name LIKE 'n1%'"));
        assertEquals(expect(i -> name(i) != null && name(i).endsWith("7")), ids("name LIKE '%7'"));
        assertEquals(expect(i -> name(i) != null && name(i).length() == 3), ids("name LIKE 'n__'"));
        assertEquals(expect(i -> name(i) != null && name(i).startsWith("n1")), ids("name ILIKE 'N1%'"));
        // NULL LIKE x and NULL NOT LIKE x are both unknown: LIKE + NOT LIKE covers exactly the non-null rows.
        int like = ids("name LIKE 'n1%'").size(), notLike = ids("name NOT LIKE 'n1%'").size();
        assertEquals(expect(i -> name(i) != null).size(), like + notLike);
        assertEquals(expect(i -> name(i) != null && name(i).matches("n[0-9]{2}")), ids("name REGEXP '^n[0-9]{2}$'"));
        assertEquals(expect(i -> name(i) != null && name(i).contains("99")), ids("name RLIKE '99'"));
    }

    @Test
    void booleanForms() throws Exception {
        assertEquals(expect(i -> Boolean.TRUE.equals(flag(i))), ids("flag"));
        assertEquals(expect(i -> Boolean.TRUE.equals(flag(i))), ids("flag = TRUE"));
        assertEquals(expect(i -> Boolean.FALSE.equals(flag(i))), ids("NOT flag"));
        assertEquals(expect(i -> Boolean.FALSE.equals(flag(i))), ids("flag = FALSE"));
        assertEquals(expect(i -> Boolean.FALSE.equals(flag(i))), ids("flag <> TRUE"), "null is unknown, not <> TRUE");
        assertEquals(expect(i -> flag(i) == null || !flag(i)), ids("flag IS NOT TRUE"), "IS NOT TRUE keeps nulls");
        assertEquals(expect(i -> flag(i) == null || flag(i)), ids("flag IS NOT FALSE"));
        assertEquals(expect(i -> Boolean.TRUE.equals(flag(i))), ids("flag IS TRUE"));
        assertEquals(expect(i -> flag(i) == null), ids("flag IS NULL"));
    }

    @Test
    void nativeAndNumericLeavesCombineUnderOrWithNulls() throws Exception {
        // i32 is null for i%10==0; the string/boolean leaves are two-valued so OR must still keep those rows.
        assertEquals(expect(i -> name(i) == null || i < 3), ids("name IS NULL OR id < 3"));
        assertEquals(expect(i -> (i % 10 != 0 && i % 100 > 90) || Boolean.TRUE.equals(flag(i))),
                ids("i32 > 90 OR flag"), "row where i32 is null but flag is true must be kept");
    }

    @Test
    void dateComparisons() throws Exception {
        String d = LocalDate.ofEpochDay(19_010).toString();
        assertEquals(expect(i -> 19_000 + i % 365 == 19_010), ids("day = DATE '" + d + "'"));
        assertEquals(expect(i -> 19_000 + i % 365 == 19_010), ids("day = '" + d + "'"), "string literal read as a date");
        assertEquals(expect(i -> 19_000 + i % 365 >= 19_010), ids("day >= DATE '" + d + "'"));
        assertEquals(expect(i -> 19_000 + i % 365 > 19_005 && 19_000 + i % 365 <= 19_010),
                ids("day BETWEEN DATE '" + LocalDate.ofEpochDay(19_006) + "' AND DATE '" + d + "'"));
    }

    private static String tsLiteral(long epochSecond) {
        return LocalDateTime.ofEpochSecond(epochSecond, 0, ZoneOffset.UTC)
                .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
    }

    @Test
    void timestampComparisonsAcrossZonesAndPrecision() throws Exception {
        long base = ParquetFixtures.TS_BASE / 1_000_000L;
        String lit = tsLiteral(base + 100);
        assertEquals(expect(i -> i >= 100), ids("ts >= TIMESTAMP '" + lit + "'"));
        assertEquals(expect(i -> i < 100), ids("ts < TIMESTAMP '" + lit + "'"));
        assertEquals(List.of(100L), ids("ts = TIMESTAMP '" + lit + "'"));
        // The fixture column is adjusted to UTC: a zoned literal is converted to the same instant.
        assertEquals(List.of(100L), ids("ts = TIMESTAMP '" + lit + "Z'"));
        assertEquals(List.of(100L), ids("ts = TIMESTAMP '" + tsLiteral(base + 100 + 7200) + "+02:00'"));
        // Half a second past the tick: never equal, and ordering stays exact.
        assertEquals(List.of(), ids("ts = TIMESTAMP '" + lit + ".5'"));
        assertEquals(expect(i -> i >= 101), ids("ts > TIMESTAMP '" + lit + ".5'"));
        assertEquals(expect(i -> i >= 10 && i <= 20),
                ids("ts BETWEEN TIMESTAMP '" + tsLiteral(base + 10) + "' AND TIMESTAMP '" + tsLiteral(base + 20) + "'"));
        assertEquals(expect(i -> i >= 100), ids("ts >= '" + lit + "'"), "string literal read as a timestamp");
    }

    @Test
    void nativePredicatesWithPassthroughOfEveryTypeAndPruning() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT name, day, ts, flag FROM t WHERE id < 200 AND name LIKE 'n1%'")) {
            List<String> names = new ArrayList<>();
            p.stream(a, b -> { for (int i = 0; i < b.getRowCount(); i++) names.add(str(b, "name", i)); });
            assertEquals(expect(i -> i < 200 && name(i) != null && name(i).startsWith("n1")).size(), names.size());
            for (String n : names) assertTrue(n.startsWith("n1"));
            assertTrue(p.lastMetrics().rowGroupsSkipped() > 0, "the numeric half of the AND must still prune: " + p.lastMetrics());
            assertTrue(p.explain().contains("native predicates"), p.explain());
        }
    }

    @Test
    void orderByAndAggregateWorkWithNativeFilters() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(),
                     "SELECT id FROM t WHERE flag IS TRUE AND name LIKE 'n%' ORDER BY id DESC LIMIT 2", a)) {
            List<Long> all = expect(i -> Boolean.TRUE.equals(flag(i)) && name(i) != null);
            assertEquals(all.get(all.size() - 1).longValue(), ((BigIntVector) r.getVector("id")).get(0));
            assertEquals(all.get(all.size() - 2).longValue(), ((BigIntVector) r.getVector("id")).get(1));
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT COUNT(*) AS c FROM t WHERE name LIKE 'n1%'", a)) {
            assertEquals((double) expect(i -> name(i) != null && name(i).startsWith("n1")).size(),
                    ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
    }

    @Test
    void typeMismatchesFailWithColumnAndType() throws Exception {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> ParquetSql.plan(main(), "SELECT id FROM t WHERE name = 5"));
        assertTrue(e.getMessage().contains("'name'") && e.getMessage().contains("STRING"), e.getMessage());
        e = assertThrows(ParquetSqlException.class, () -> ParquetSql.plan(main(), "SELECT id FROM t WHERE id = 'x'"));
        assertTrue(e.getMessage().contains("'id'"), e.getMessage());
        e = assertThrows(ParquetSqlException.class, () -> ParquetSql.plan(main(), "SELECT id FROM t WHERE name * 2 > 1"));
        assertTrue(e.getMessage().contains("'name'") && e.getMessage().contains("STRING"), e.getMessage());
        e = assertThrows(ParquetSqlException.class, () -> ParquetSql.plan(main(), "SELECT id FROM t WHERE day = 'yesterday'"));
        assertTrue(e.getMessage().contains("DATE"), e.getMessage());
    }

    @Test
    void int64BeyondTwoToTheFiftyThreeIsNeverRoundedSilently() throws Exception {
        Path f = Files.createTempDirectory("big").resolve("b.parquet");
        ParquetFixtures.writeBigInt64(f);
        try (BufferAllocator a = new RootAllocator()) {
            // Passthrough is exact: no widening involved.
            try (VectorSchemaRoot r = ParquetSql.execute(f, "SELECT big FROM t WHERE id = 2", a)) {
                assertEquals(9_007_199_254_740_993L, ((BigIntVector) r.getVector("big")).get(0));
            }
            // Arithmetic needs the double view: refuse, naming the column.
            ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> {
                try (VectorSchemaRoot r = ParquetSql.execute(f, "SELECT big + 1 AS b FROM t", a)) { }
            });
            assertTrue(e.getMessage().contains("'big'") && e.getMessage().contains("2^53"), e.getMessage());
            // Explicit opt-in rounds.
            try (ParquetSqlPlan p = ParquetSql.plan(f, "SELECT big + 1 AS b FROM t",
                    ParquetSqlOptions.defaults().withInt64Policy(ParquetSqlOptions.Int64Policy.ROUND_TO_DOUBLE));
                 VectorSchemaRoot r = p.execute(a)) {
                assertEquals(3, r.getRowCount());
                assertEquals(1L, p.roundedValues());
            }
        }
    }

    // ---------------------------------------------------------------- FROM resolution

    @Test
    void fromResolutionRules() throws Exception {
        Path m = main();
        ParquetSql sql = ParquetSql.create().register("events", m);
        try (BufferAllocator a = new RootAllocator()) {
            try (VectorSchemaRoot r = sql.execute("SELECT COUNT(*) AS n FROM events", a)) {
                assertEquals((double) ROWS, ((Float8Vector) r.getVector("n")).get(0), 0.0);
            }
            try (VectorSchemaRoot r = sql.execute("SELECT COUNT(*) AS n FROM '" + m + "'", a)) {
                assertEquals((double) ROWS, ((Float8Vector) r.getVector("n")).get(0), 0.0);
            }
            assertThrows(ParquetSqlException.class, () -> sql.plan("SELECT id FROM nope"));
            // Explicit file: the bare name is ignored, as in the first bridge.
            try (VectorSchemaRoot r = ParquetSql.execute(m, "SELECT id FROM anything WHERE id < 2", a)) {
                assertEquals(2, r.getRowCount());
            }
            assertThrows(ParquetSqlException.class, () -> ParquetSql.plan(m, "SELECT id FROM '" + m + "'"));
        }
    }

    @Test
    void globReadsSeveralFilesAndChecksSchemas() throws Exception {
        Path d = Files.createTempDirectory("multi");
        ParquetFixtures.writeMain(d.resolve("a.parquet"), 1000);
        ParquetFixtures.writeMain(d.resolve("b.parquet"), 500);
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.create().execute(
                     "SELECT COUNT(*) AS n FROM read_parquet('" + d + "/*.parquet')", a)) {
            assertEquals(1500.0, ((Float8Vector) r.getVector("n")).get(0), 0.0);
        }
        ParquetFixtures.writeBigInt64(d.resolve("c.parquet"));
        try (BufferAllocator a = new RootAllocator()) {
            ParquetSqlException e = assertThrows(ParquetSqlException.class,
                    () -> ParquetSql.create().plan("SELECT id FROM read_parquet('" + d + "/*.parquet')"));
            assertTrue(e.getMessage().toLowerCase().contains("mismatch"), e.getMessage());
        }
    }

    @Test
    void allowedRootRejectsPathsOutsideIt() throws Exception {
        Path m = main();
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withAllowedRoot(Files.createTempDirectory("jail"));
        ParquetSqlException e = assertThrows(ParquetSqlException.class,
                () -> ParquetSql.create(o).plan("SELECT id FROM '" + m + "'"));
        assertTrue(e.getMessage().contains("outside the allowed root"), e.getMessage());
    }

    // ---------------------------------------------------------------- EXPLAIN

    @Test
    void explainDescribesThePlan() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "EXPLAIN SELECT id FROM t WHERE id < 10 ORDER BY id", a)) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < r.getRowCount(); i++) sb.append(str(r, "plan", i)).append('\n');
            String s = sb.toString();
            assertTrue(s.contains("mode: materialize"), s);
            assertTrue(s.contains("pruning:"), s);
            assertTrue(s.contains("columns decoded:"), s);
        }
    }

    // ---------------------------------------------------------------- bounded top-N

    /** i32 of fixture row i, or null: the fixture writes null when i % 10 == 0. */
    private static Integer i32(long i) { return i % 10 == 0 ? null : (Integer) (int) (i % 100); }

    /** Expected ids of {@code ORDER BY i32 [DESC], id [DESC] LIMIT n}: nulls last, stable by id for ties. */
    private static List<Long> topByI32(boolean desc, boolean idDesc, int n) {
        List<Long> ids = new ArrayList<>();
        for (long i = 0; i < ROWS; i++) ids.add(i);
        ids.sort((x, y) -> {
            Integer a = i32(x), b = i32(y);
            int c;
            if (a == null || b == null) c = a == null ? (b == null ? 0 : 1) : -1;
            else c = desc ? Integer.compare(b, a) : Integer.compare(a, b);
            if (c == 0) c = idDesc ? Long.compare(y, x) : Long.compare(x, y);
            return c;
        });
        return ids.subList(0, n);
    }

    private List<Long> runIds(String sql, ParquetSqlOptions o) throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), sql, o);
             VectorSchemaRoot r = p.execute(a)) {
            List<Long> out = new ArrayList<>();
            BigIntVector id = (BigIntVector) r.getVector("id");
            for (int k = 0; k < r.getRowCount(); k++) out.add(id.get(k));
            return out;
        }
    }

    @Test
    void boundedTopNMatchesAFullSortAcrossManyBatches() throws Exception {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withBatchSize(1000);
        assertEquals(topByI32(true, false, 25), runIds("SELECT id, i32 FROM t ORDER BY i32 DESC LIMIT 25", o), "desc, ties keep file order");
        assertEquals(topByI32(false, false, 25), runIds("SELECT id, i32 FROM t ORDER BY i32 LIMIT 25", o), "asc, nulls last");
        assertEquals(topByI32(true, true, 4), runIds("SELECT id FROM t ORDER BY i32 DESC, id DESC LIMIT 4", o), "two keys");
    }

    @Test
    void boundedTopNCompactsWhenTheLimitIsLargeAndKeepsNullsLast() throws Exception {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withBatchSize(1000);
        assertEquals(topByI32(true, false, 3000), runIds("SELECT id FROM t ORDER BY i32 DESC LIMIT 3000", o), "compaction");
        assertEquals(topByI32(false, false, 19000), runIds("SELECT id FROM t ORDER BY i32 LIMIT 19000", o), "includes the null tail");
    }

    @Test
    void boundedTopNOrdersByAComputedAlias() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT id, i32 + 1 AS p FROM t ORDER BY p DESC LIMIT 5", a)) {
            assertEquals(5, r.getRowCount());
            assertEquals(2, r.getFieldVectors().size(), "hidden sort keys must not leak into the result");
            for (int k = 0; k < 5; k++) {
                assertEquals(99L + 100L * k, ((BigIntVector) r.getVector("id")).get(k));
                assertEquals(100.0, ((Float8Vector) r.getVector("p")).get(k), 0.0);
            }
        }
    }

    @Test
    void boundedTopNEmptyResultKeepsTheColumns() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = ParquetSql.execute(main(), "SELECT id, name FROM t WHERE id < 0 ORDER BY id LIMIT 3", a)) {
            assertEquals(0, r.getRowCount());
            assertNotNull(r.getVector("id"));
            assertNotNull(r.getVector("name"));
        }
    }

    @Test
    void explainReportsBoundedTopN() throws Exception {
        try (ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t ORDER BY id DESC LIMIT 10")) {
            assertTrue(p.explain().contains("bounded top-10"), p.explain());
        }
        try (ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t ORDER BY id DESC")) {
            assertTrue(p.explain().contains("materialize"), "no LIMIT: every row is needed\n" + p.explain());
        }
    }
}
