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

    private static final int AWKWARD_ROWS = 100;
    private static Path awkward;

    private static synchronized Path awkward() throws IOException {
        if (awkward == null) {
            main();
            awkward = dir.resolve("awkward.parquet");
            ParquetFixtures.writeAwkward(awkward, AWKWARD_ROWS);
        }
        return awkward;
    }

    private static final int AWKWARD_LARGE_ROWS = 5_000;
    private static Path awkwardLarge;

    /** The same columns over many row groups, for pruning. */
    private static synchronized Path awkwardLarge() throws IOException {
        if (awkwardLarge == null) {
            main();
            awkwardLarge = dir.resolve("awkward-large.parquet");
            ParquetFixtures.writeAwkward(awkwardLarge, AWKWARD_LARGE_ROWS);
        }
        return awkwardLarge;
    }

    private VectorSchemaRoot runAwkward(BufferAllocator a, String sql) throws Exception {
        return ParquetSql.execute(awkward(), sql, a);
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

    // ---------------------------------------------------------------- DISTINCT and non-numeric aggregates

    private VectorSchemaRoot runSql(BufferAllocator a, String sql) throws Exception {
        return ParquetSql.execute(main(), sql, a);
    }

    @Test
    void countOfAColumnCountsNonNullsForAnyType() throws Exception {
        long names = 0, flags = 0;
        for (int i = 0; i < ROWS; i++) { if (i % 11 != 0) names++; if (i % 7 != 0) flags++; }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*), COUNT(name), COUNT(flag), COUNT(day) FROM t")) {
            assertEquals(1, r.getRowCount());
            assertEquals(ROWS, ((Float8Vector) r.getVector("COUNT(*)")).get(0), 0.0);
            assertEquals(names, ((Float8Vector) r.getVector("COUNT(name)")).get(0), 0.0);
            assertEquals(flags, ((Float8Vector) r.getVector("COUNT(flag)")).get(0), 0.0);
            assertEquals(ROWS, ((Float8Vector) r.getVector("COUNT(day)")).get(0), 0.0);
        }
    }

    @Test
    void countDistinctOverStringAndDateAndNumeric() throws Exception {
        java.util.Set<String> names = new java.util.HashSet<>();
        java.util.Set<Integer> days = new java.util.HashSet<>(), i32s = new java.util.HashSet<>();
        for (int i = 0; i < ROWS; i++) {
            if (i % 11 != 0) names.add("n" + i);
            days.add(19000 + i % 365);
            if (i % 10 != 0) i32s.add(i % 100);
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(DISTINCT name) AS dn, COUNT(DISTINCT day) AS dd, COUNT(DISTINCT i32) AS di FROM t")) {
            assertEquals(names.size(), ((Float8Vector) r.getVector("dn")).get(0), 0.0);
            assertEquals(days.size(), ((Float8Vector) r.getVector("dd")).get(0), 0.0);
            assertEquals(i32s.size(), ((Float8Vector) r.getVector("di")).get(0), 0.0);
        }
    }

    @Test
    void minMaxOverStringDateTimestampKeepTheColumnType() throws Exception {
        String lo = null, hi = null;
        for (int i = 0; i < ROWS; i++) {
            if (i % 11 == 0) continue;
            String n = "n" + i;
            if (lo == null || n.compareTo(lo) < 0) lo = n;
            if (hi == null || n.compareTo(hi) > 0) hi = n;
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT MIN(name) AS lo, MAX(name) AS hi, MIN(day) AS d0, MAX(day) AS d1, MAX(ts) AS t1, MIN(id) AS i0 FROM t")) {
            assertEquals(lo, new String(((VarCharVector) r.getVector("lo")).get(0), StandardCharsets.UTF_8));
            assertEquals(hi, new String(((VarCharVector) r.getVector("hi")).get(0), StandardCharsets.UTF_8));
            assertEquals(19000, ((DateDayVector) r.getVector("d0")).get(0));
            assertEquals(19000 + 364, ((DateDayVector) r.getVector("d1")).get(0));
            assertEquals(ParquetFixtures.TS_BASE + (ROWS - 1) * 1_000_000L, ((TimeStampVector) r.getVector("t1")).get(0));
            assertEquals(0.0, ((Float8Vector) r.getVector("i0")).get(0), 0.0);
        }
    }

    @Test
    void foldedAggregatesHonourTheWhereAndPruneRowGroups() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT COUNT(name), MAX(name) FROM t WHERE id >= 19000");
             VectorSchemaRoot r = p.execute(a)) {
            long n = 0;
            for (int i = 19000; i < ROWS; i++) if (i % 11 != 0) n++;
            assertEquals(n, ((Float8Vector) r.getVector("COUNT(name)")).get(0), 0.0);
            assertNotNull(p.lastMetrics());
            assertTrue(p.lastMetrics().rowGroupsSkipped() > 0, "WHERE id >= 19000 must skip row groups");
        }
    }

    @Test
    void aggregatesOverNoRowsAreZeroAndNull() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(name) AS c, MAX(name) AS m, COUNT(DISTINCT day) AS d FROM t WHERE id < 0")) {
            assertEquals(1, r.getRowCount());
            assertEquals(0.0, ((Float8Vector) r.getVector("c")).get(0), 0.0);
            assertTrue(r.getVector("m").isNull(0));
            assertEquals(0.0, ((Float8Vector) r.getVector("d")).get(0), 0.0);
        }
    }

    @Test
    void selectDistinctReturnsEachValueOnceIncludingNull() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT DISTINCT flag FROM t")) {
            assertEquals(3, r.getRowCount(), "true, false and null");
            java.util.Set<Object> seen = new java.util.HashSet<>();
            for (int i = 0; i < 3; i++) seen.add(r.getVector("flag").getObject(i));
            assertEquals(3, seen.size());
            assertTrue(seen.contains(null) && seen.contains(Boolean.TRUE) && seen.contains(Boolean.FALSE));
        }
    }

    @Test
    void selectDistinctWithOrderByAndLimit() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT DISTINCT day FROM t ORDER BY day LIMIT 3")) {
            assertEquals(3, r.getRowCount());
            for (int i = 0; i < 3; i++) assertEquals(19000 + i, ((DateDayVector) r.getVector("day")).get(i));
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT DISTINCT i32 AS v FROM t ORDER BY v DESC LIMIT 2")) {
            assertEquals(2, r.getRowCount());
            assertEquals(99, ((IntVector) r.getVector("v")).get(0));
            assertEquals(98, ((IntVector) r.getVector("v")).get(1));
        }
    }

    @Test
    void selectDistinctLimitWithoutOrderByStopsEarlyAndCountsDistinct() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT DISTINCT day FROM t LIMIT 5");
             VectorSchemaRoot r = p.execute(a)) {
            assertEquals(5, r.getRowCount());
            java.util.Set<Object> seen = new java.util.HashSet<>();
            for (int i = 0; i < 5; i++) seen.add(r.getVector("day").getObject(i));
            assertEquals(5, seen.size());
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT DISTINCT day, flag FROM t WHERE id < 0")) {
            assertEquals(0, r.getRowCount());
            assertNotNull(r.getVector("day"));
        }
    }

    @Test
    void foldedQueriesRejectWhatTheyCannotDoWithNamedErrors() throws Exception {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runSql(a, "SELECT flag, COUNT(DISTINCT name) FROM t GROUP BY flag")) { }
        });
        assertTrue(e.getMessage().contains("GROUP BY"), e.getMessage());
        e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runSql(a, "SELECT COUNT(DISTINCT name), SUM(id + 1) FROM t")) { }
        });
        assertTrue(e.getMessage().contains("plain column"), e.getMessage());
        e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runSql(a, "SELECT DISTINCT id FROM t ORDER BY i32")) { }
        });
        assertTrue(e.getMessage().contains("select-list"), e.getMessage());
        assertThrows(UnsupportedOperationException.class, () -> {
            try (ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT DISTINCT day FROM t")) {
                p.stream(new RootAllocator(), x -> { });
            }
        });
    }

    // ---------------------------------------------------------------- OFFSET

    @Test
    void offsetWithOrderByAndLimitWindowsTheSortedRows() throws Exception {
        assertEquals(List.of(10L, 11L, 12L, 13L, 14L), runIds("SELECT id FROM t ORDER BY id LIMIT 5 OFFSET 10", ParquetSqlOptions.defaults().withBatchSize(1000)));
        // descending order is 19999 ... 0, so skipping 19999 rows leaves only id 0
        assertEquals(List.of(0L), runIds("SELECT id FROM t ORDER BY id DESC LIMIT 3 OFFSET 19999", ParquetSqlOptions.defaults()));
        assertEquals(List.of(19997L, 19996L), runIds("SELECT id FROM t ORDER BY id DESC LIMIT 2 OFFSET 2", ParquetSqlOptions.defaults()));
        assertEquals(List.of(), runIds("SELECT id FROM t ORDER BY id LIMIT 3 OFFSET 20000", ParquetSqlOptions.defaults()));
    }

    @Test
    void offsetWithoutOrderByOrLimit() throws Exception {
        assertEquals(List.of(102L, 103L, 104L), runIds("SELECT id FROM t WHERE id >= 100 LIMIT 3 OFFSET 2", ParquetSqlOptions.defaults()));
        assertEquals(List.of(19998L, 19999L), runIds("SELECT id FROM t OFFSET 19998", ParquetSqlOptions.defaults()));
        assertEquals(List.of(), runIds("SELECT id FROM t OFFSET 20000", ParquetSqlOptions.defaults()));
    }

    @Test
    void offsetAlsoAppliesWhenStreaming() throws Exception {
        List<Long> got = new ArrayList<>();
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t LIMIT 2500 OFFSET 1500", ParquetSqlOptions.defaults().withBatchSize(1000))) {
            p.stream(a, root -> {
                BigIntVector id = (BigIntVector) root.getVector("id");
                for (int i = 0; i < root.getRowCount(); i++) got.add(id.get(i));
            });
        }
        assertEquals(2500, got.size());
        assertEquals(1500L, got.get(0));
        assertEquals(3999L, got.get(got.size() - 1));
    }

    @Test
    void offsetOnASingleRowAggregateYieldsNoRows() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) FROM t OFFSET 1")) {
            assertEquals(0, r.getRowCount());
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) FROM t LIMIT 1 OFFSET 0")) {
            assertEquals(1, r.getRowCount());
        }
    }

    @Test
    void offsetWithDistinct() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT DISTINCT day FROM t ORDER BY day LIMIT 3 OFFSET 2")) {
            assertEquals(3, r.getRowCount());
            for (int i = 0; i < 3; i++) assertEquals(19002 + i, ((DateDayVector) r.getVector("day")).get(i));
        }
    }

    // ---------------------------------------------------------------- COALESCE / NULLIF

    @Test
    void coalesceInTheSelectListKeepsNullSemanticsAndNames() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT id, COALESCE(i32, -1) AS v, COALESCE(i32, 0) + 1 FROM t WHERE id < 25 ORDER BY id")) {
            assertEquals(25, r.getRowCount());
            Float8Vector v = (Float8Vector) r.getVector("v");
            Float8Vector w = (Float8Vector) r.getVector("COALESCE(i32, 0) + 1");
            for (int i = 0; i < 25; i++) {
                double expected = i % 10 == 0 ? -1 : i % 100;
                assertEquals(expected, v.get(i), 0.0, "row " + i);
                assertEquals((i % 10 == 0 ? 0 : i % 100) + 1, w.get(i), 0.0, "row " + i);
            }
        }
    }

    @Test
    void coalesceAndNullifInWhere() throws Exception {
        long zero = 0, kept = 0;
        for (int i = 0; i < ROWS; i++) {
            if (i % 10 == 0) zero++;                       // null i32 coalesces to 0
            if (i % 10 != 0 && i % 100 != 5 && i % 100 > 0) kept++; // NULLIF(i32, 5) > 0
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) AS c FROM t WHERE COALESCE(i32, 0) = 0")) {
            assertEquals(zero, ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) AS c FROM t WHERE NULLIF(i32, 5) > 0")) {
            assertEquals(kept, ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
    }

    @Test
    void coalesceInsideAggregatesAndOrderBy() throws Exception {
        double sum = 0;
        for (int i = 0; i < ROWS; i++) if (i % 10 != 0) sum += i % 100;
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT SUM(COALESCE(i32, 0)) AS s FROM t")) {
            assertEquals(sum, ((Float8Vector) r.getVector("s")).get(0), 0.0);
        }
        // nulls coalesce to -1 so they sort first ascending; ties keep file order
        assertEquals(List.of(0L, 10L, 20L), runIds("SELECT id FROM t ORDER BY COALESCE(i32, -1) LIMIT 3", ParquetSqlOptions.defaults().withBatchSize(1000)));
    }

    @Test
    void coalesceUnderOrKeepsTheOtherBranch() throws Exception {
        List<Long> expect = new ArrayList<>();
        for (long i = 0; i < ROWS; i++) if ((i % 10 == 0 ? 0 : i % 100) > 98 || i < 3) expect.add(i);
        assertEquals(expect, runIds("SELECT id FROM t WHERE COALESCE(i32, 0) > 98 OR id < 3", ParquetSqlOptions.defaults()));
    }

    @Test
    void coalesceOverAStringColumnFailsWithANamedError() throws Exception {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runSql(a, "SELECT COALESCE(name, 'x') FROM t")) { }
        });
        assertTrue(e.getMessage().contains("STRING"), e.getMessage());
    }

    // ---------------------------------------------------------------- string functions

    private static String nameOf(int i) { return i % 11 == 0 ? null : "n" + i; }

    @Test
    void upperInTheSelectListNullInNullOut() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT id, UPPER(name) AS u, LOWER(name) || '!' FROM t WHERE id < 30 ORDER BY id")) {
            assertEquals(30, r.getRowCount());
            for (int i = 0; i < 30; i++) {
                String n = nameOf(i);
                assertEquals(n == null ? null : n.toUpperCase(), str(r, "u", i), "row " + i);
                assertEquals(n == null ? null : n + "!", str(r, "LOWER(name) || '!'", i), "row " + i);
            }
        }
    }

    @Test
    void stringFunctionsInWhereUseTheNativeStringKernels() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) AS c FROM t WHERE UPPER(name) = 'N5'")) {
            assertEquals(1.0, ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
        long like = 0, sub = 0;
        for (int i = 0; i < ROWS; i++) {
            String n = nameOf(i);
            if (n != null && n.startsWith("n1")) like++;
            if (n != null && n.length() > 1 && n.charAt(1) == '1') sub++;
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) AS c FROM t WHERE UPPER(name) LIKE 'N1%'")) {
            assertEquals(like, ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) AS c FROM t WHERE SUBSTRING(name, 2, 1) = '1'")) {
            assertEquals(sub, ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
    }

    @Test
    void lengthIsNumericSoItWorksInArithmeticAggregatesAndOrderBy() throws Exception {
        long three = 0;
        double sum = 0;
        for (int i = 0; i < ROWS; i++) {
            String n = nameOf(i);
            if (n == null) continue;
            sum += n.length();
            if (n.length() == 3) three++;
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT COUNT(*) AS c FROM t WHERE LENGTH(name) = 3")) {
            assertEquals(three, ((Float8Vector) r.getVector("c")).get(0), 0.0);
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT SUM(LENGTH(name)) AS s FROM t")) {
            assertEquals(sum, ((Float8Vector) r.getVector("s")).get(0), 0.0);
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT id, LENGTH(name) + 1 AS l FROM t WHERE id < 12 ORDER BY id")) {
            assertEquals(12, r.getRowCount());
            assertTrue(r.getVector("l").isNull(0), "LENGTH of a null name is null");
            assertEquals(3.0, ((Float8Vector) r.getVector("l")).get(1), 0.0, "row 1 is n1: length 2, plus 1");
        }
        // names with 6 characters are ids 10000.., ties keep file order
        assertEquals(List.of(10000L, 10001L, 10002L),
                runIds("SELECT id FROM t WHERE name IS NOT NULL ORDER BY LENGTH(name) DESC LIMIT 3", ParquetSqlOptions.defaults().withBatchSize(1000)));
    }

    @Test
    void stringPredicateUnderOrKeepsTheOtherBranch() throws Exception {
        assertEquals(List.of(0L, 1L, 2L, 5L), runIds("SELECT id FROM t WHERE UPPER(name) = 'N5' OR id < 3", ParquetSqlOptions.defaults()));
    }

    @Test
    void aProjectionOfOnlyDerivedStringsWorksAndStreams() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runSql(a, "SELECT UPPER(name) FROM t LIMIT 3")) {
            assertEquals(3, r.getRowCount());
            assertTrue(r.getVector("UPPER(name)").isNull(0));
            assertEquals("N1", str(r, "UPPER(name)", 1));
            assertEquals("N2", str(r, "UPPER(name)", 2));
        }
        List<String> got = new ArrayList<>();
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id, TRIM(name || ' ') AS t FROM t LIMIT 2500 OFFSET 1500", ParquetSqlOptions.defaults().withBatchSize(1000))) {
            p.stream(a, root -> {
                for (int i = 0; i < root.getRowCount(); i++) got.add(str(root, "t", i));
            });
        }
        assertEquals(2500, got.size());
        assertEquals(nameOf(1500), got.get(0));
        assertEquals(nameOf(3999), got.get(got.size() - 1));
    }

    @Test
    void stringFunctionsRejectWhatTheyCannotDoWithNamedErrors() throws Exception {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runSql(a, "SELECT id FROM t ORDER BY LOWER(name) LIMIT 3")) { }
        });
        assertTrue(e.getMessage().contains("ORDER BY"), e.getMessage());
        e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runSql(a, "SELECT LOWER(id) FROM t")) { }
        });
        assertTrue(e.getMessage().contains("INT64"), e.getMessage());
    }

    @Test
    void explainListsTheDerivedColumns() throws Exception {
        try (ParquetSqlPlan p = ParquetSql.plan(main(), "SELECT id FROM t WHERE LENGTH(name) > 3 AND UPPER(name) = 'N5'")) {
            String ex = p.explain();
            assertTrue(ex.contains("derived column"), ex);
            assertTrue(ex.contains("derived string columns"), ex);
        }
    }

    // ---------------------------------------------------------------- quoted identifiers

    @Test
    void quotedColumnsAreSelectedUnderTheirRealNames() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT \"my col\", \"1st\", `a-b` FROM t WHERE id < 3 ORDER BY id")) {
            assertEquals(3, r.getRowCount());
            assertEquals(2, ((IntVector) r.getVector("my col")).get(1));
            assertEquals("v2", str(r, "1st", 2));
            assertEquals(2, ((IntVector) r.getVector("a-b")).get(2));
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT * FROM t WHERE id = 4")) {
            assertEquals(5, r.getFieldVectors().size());
            assertEquals(8, ((IntVector) r.getVector("my col")).get(0));
        }
    }

    @Test
    void quotedNumericColumnsWorkInExpressionsWhereAndAggregates() throws Exception {
        double sum = 0;
        for (int i = 0; i < AWKWARD_ROWS; i++) sum += 2 * i;
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT SUM(\"my col\") AS s, MAX(\"select\") AS m FROM t")) {
            assertEquals(sum, ((Float8Vector) r.getVector("s")).get(0), 0.0);
            assertEquals(990.0, ((Float8Vector) r.getVector("m")).get(0), 0.0);
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT \"select\" + 1 AS x, \"my col\" * 2 FROM t WHERE id = 2")) {
            assertEquals(21.0, ((Float8Vector) r.getVector("x")).get(0), 0.0);
            assertEquals(8.0, ((Float8Vector) r.getVector("my col * 2")).get(0), 0.0, "the default output name shows the real name");
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT id FROM t WHERE \"my col\" > 190")) {
            assertEquals(4, r.getRowCount());
        }
    }

    @Test
    void quotedColumnsPruneRowGroupsAndSortAndLimit() throws Exception {
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT id FROM t WHERE \"my col\" >= 190")) {
            assertEquals(5, r.getRowCount());
        }
        // "my col" = 2 * id, so >= 9990 keeps ids 4995..4999 and the earlier row groups can be skipped
        try (BufferAllocator a = new RootAllocator();
             ParquetSqlPlan p = ParquetSql.plan(awkwardLarge(), "SELECT id FROM t WHERE \"my col\" >= 9990");
             VectorSchemaRoot r = p.execute(a)) {
            assertEquals(5, r.getRowCount());
            assertTrue(p.lastMetrics().rowGroupsInFile() > 1, "the fixture must have several row groups: " + p.lastMetrics());
            assertTrue(p.lastMetrics().rowGroupsSkipped() > 0, "pushdown must use the real column name: " + p.lastMetrics());
        }
        assertEquals(List.of(99L, 98L), runAwkwardIds("SELECT id FROM t ORDER BY \"my col\" DESC LIMIT 2"));
    }

    private List<Long> runAwkwardIds(String sql) throws Exception {
        try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runAwkward(a, sql)) {
            List<Long> out = new ArrayList<>();
            for (int i = 0; i < r.getRowCount(); i++) out.add(((BigIntVector) r.getVector("id")).get(i));
            return out;
        }
    }

    @Test
    void quotedStringColumnsWorkInNativePredicatesFoldsAndStringFunctions() throws Exception {
        assertEquals(List.of(7L), runAwkwardIds("SELECT id FROM t WHERE \"1st\" = 'v7'"));
        assertEquals(11, runAwkwardIds("SELECT id FROM t WHERE \"1st\" LIKE 'v1%'").size(), "v1, v10..v19 and nothing else below 100");
        long threes = 0;
        for (int i = 0; i < AWKWARD_ROWS; i++) if (i % 7 == 3) threes++;
        assertEquals(threes, runAwkwardIds("SELECT id FROM t WHERE `a-b` = 3").size());
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT COUNT(DISTINCT \"1st\") AS d, MAX(\"1st\") AS m FROM t")) {
            assertEquals(100.0, ((Float8Vector) r.getVector("d")).get(0), 0.0);
            assertEquals("v99", new String(((VarCharVector) r.getVector("m")).get(0), StandardCharsets.UTF_8));
        }
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT UPPER(\"1st\") AS u, LENGTH(\"1st\") AS l FROM t WHERE id = 42")) {
            assertEquals("V42", str(r, "u", 0));
            assertEquals(3.0, ((Float8Vector) r.getVector("l")).get(0), 0.0);
        }
    }

    @Test
    void anUnknownQuotedColumnOrAQuotedAliasReferenceFailsWithTheRealName() throws Exception {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runAwkward(a, "SELECT \"no such\" FROM t")) { }
        });
        assertTrue(e.getMessage().contains("no such"), e.getMessage());
        e = assertThrows(ParquetSqlException.class, () -> {
            try (BufferAllocator a = new RootAllocator(); VectorSchemaRoot r = runAwkward(a, "SELECT id AS \"my alias\" FROM t ORDER BY \"my alias\"")) { }
        });
        assertTrue(e.getMessage().contains("my alias"), e.getMessage());
        try (BufferAllocator a = new RootAllocator();
             VectorSchemaRoot r = runAwkward(a, "SELECT id AS \"my alias\" FROM t WHERE id < 2")) {
            assertNotNull(r.getVector("my alias"), "a quoted alias can name an output");
        }
    }
}
