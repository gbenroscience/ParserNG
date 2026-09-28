package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VarCharVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import static com.github.gbenroscience.parser.ng.parquet.v1.ParquetTestData.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * exactFilter() turns pushdown from a pruning hint into a real row filter. Each test compares the scan's output
 * against a plain-Java oracle over the same rows, so the expectation is the SQL semantics, not a copy of the code.
 */
class ExactFilterTest {

    @TempDir
    static Path dir;
    static Path flat;

    @BeforeAll
    static void writeFixture() throws IOException {
        flat = ParquetTestData.writeFlat(dir, "flat.parquet");
    }

    private static final String[] ALL = FLAT_COLUMNS.toArray(new String[0]);

    /** Runs the predicate in exact mode, sequential and parallel, and checks both equal the oracle. */
    private static void check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate p, Predicate<Row> oracle) {
        List<Long> expected = expectedIds(oracle);
        ParquetScan s = ParquetScan.scan(flat).select(ALL).pushdown(p).exactFilter();
        assertEquals(expected, ids(s), "sequential: " + p);
        assertEquals(expected, ids(s.parallelism(3)), "parallel: " + p);
    }

    private static boolean eq(Object a, Object b) { return a != null && a.equals(b); }

    // ================================================================ comparisons

    @Test
    void numericComparisonsExcludeNullsForEveryOperator() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("grp", 3), r -> eq(r.grp(), 3));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ne("grp", 3), r -> r.grp() != null && r.grp() != 3);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("grp", 3), r -> r.grp() != null && r.grp() < 3);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.le("grp", 3), r -> r.grp() != null && r.grp() <= 3);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.gt("grp", 3), r -> r.grp() != null && r.grp() > 3);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("grp", 3), r -> r.grp() != null && r.grp() >= 3);
    }

    @Test
    void int64AcceptsIntegerAndLongLiterals() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.gt("id", 4_990L), r -> r.id() > 4_990L);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.gt("id", 4_990), r -> r.id() > 4_990);
    }

    @Test
    void doubleColumnAcceptsDoubleFloatAndIntegerLiterals() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("score", 2_000.0d), r -> r.score() >= 2_000.0);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("score", 2_000.0f), r -> r.score() >= 2_000.0);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("score", 2_000), r -> r.score() >= 2_000.0);
    }

    @Test
    void floatColumn() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("ratio", 2.0f), r -> r.ratio() != null && r.ratio() == 2.0f);
    }

    @Test
    void booleanColumnEquality() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("flag", true), Row::flag);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ne("flag", true), r -> !r.flag());
    }

    @Test
    void dateColumnTakesLocalDate() {
        LocalDate cut = LocalDate.ofEpochDay(DAY_BASE + 10);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("day", cut), r -> r.day() != null && r.day() >= DAY_BASE + 10);
    }

    @Test
    void timestampColumnTakesLong() {
        long cut = TS_BASE + 2_000_000L;
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("ts", cut), r -> r.ts() != null && r.ts() < cut);
    }

    /**
     * Regression: DATE and TIMESTAMP columns are DateDayVector / TimeStamp*Vector in Arrow, which are NOT subclasses of
     * IntVector / BigIntVector. The IN and column-to-column nodes had the same bad cast as the Cmp node, and only the
     * Cmp node was covered. ("day" and "ts" are the only DATE/TIMESTAMP columns, so ColCmp compares each with itself:
     * enough to drive the type-specific read path for both operands.)
     */
    @Test
    void dateAndTimestampColumnsWorkInInAndColumnComparisons() {
        LocalDate d3 = LocalDate.ofEpochDay(DAY_BASE + 3), d7 = LocalDate.ofEpochDay(DAY_BASE + 7);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.in("day", d3, d7),
                r -> r.day() != null && (r.day() == DAY_BASE + 3 || r.day() == DAY_BASE + 7));
        long t1 = TS_BASE + 1_000L, t5 = TS_BASE + 5_000L;
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.in("ts", t1, t5),
                r -> r.ts() != null && (r.ts() == t1 || r.ts() == t5));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colEq("day", "day"), r -> r.day() != null);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colLt("day", "day"), r -> false);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colGe("ts", "ts"), r -> r.ts() != null);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colLt("ts", "ts"), r -> false);
    }

    @Test
    void stringEqualityAndRangeAreExactOnDecodedStrings() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("name", "gamma"), r -> eq(r.name(), "gamma"));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("name", "delta"), r -> r.name() != null && r.name().compareTo("delta") >= 0);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("name", "b"), r -> r.name() != null && r.name().compareTo("b") < 0);
    }

    // ================================================================ IN / IS NULL

    @Test
    void inList() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.in("grp", 1, 4, 7), r -> r.grp() != null && Set.of(1, 4, 7).contains(r.grp()));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.in("name", "alpha", "gamma"), r -> r.name() != null && Set.of("alpha", "gamma").contains(r.name()));
    }

    @Test
    void isNullAndIsNotNull() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.isNull("grp"), r -> r.grp() == null);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.isNotNull("name"), r -> r.name() != null);
    }

    // ================================================================ boolean structure

    @Test
    void andOrCompose() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.and(
                        com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("id", 1_000L),
                        com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("id", 1_100L)),
                r -> r.id() >= 1_000 && r.id() < 1_100);
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.or(
                        com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("id", 10L),
                        com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("id", 4_990L)),
                r -> r.id() < 10 || r.id() >= 4_990);
    }

    @Test
    void notIsPlainBooleanNegationSoNullRowsComeBackIn() {
        // Cmp on a null is "unknown -> false" at the leaf; Not then flips that, which is this module's documented behaviour.
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.not(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("grp", 3)),
                r -> !(r.grp() != null && r.grp() == 3));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.not(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.isNull("grp")), r -> r.grp() != null);
    }

    // ================================================================ ColCmp / LIKE / regex

    @Test
    void columnToColumnComparison() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colLt("lo", "hi"), r -> r.lo() < r.hi());
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colEq("lo", "hi"), r -> r.lo() == r.hi());
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colGe("lo", "hi"), r -> r.lo() >= r.hi());
    }

    @Test
    void likeAndIlike() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.like("name", "alpha%"), r -> r.name() != null && r.name().startsWith("alpha"));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.like("name", "alpha"), r -> eq(r.name(), "alpha"));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ilike("name", "BETA"), r -> r.name() != null && r.name().equalsIgnoreCase("beta"));
        // Unescaped '_' is the single-character wildcard (SQL LIKE), so it matches BOTH "delta_1" and "deltaX1".
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.like("name", "delta_1"), r -> eq(r.name(), "delta_1") || eq(r.name(), "deltaX1"));
        // Escaped '\_' is a literal underscore, so only "delta_1" matches.
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.like("name", "delta\\_1"), r -> eq(r.name(), "delta_1"));
    }

    @Test
    void regexUsesFindSemantics() {
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.regex("name", "lph"), r -> r.name() != null && r.name().contains("lph"));
        check(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.regex("name", "^a.*t$"), r -> r.name() != null && r.name().matches("a.*t"));
    }

    // ================================================================ custom

    @Test
    void customPredicateIsInvokedPerSurvivingRowAndOwnsItsNullHandling() {
        CustomPredicate evenScoreId = (batch, row) -> {
            BigIntVector id = (BigIntVector) batch.getVector("id");
            return id.get(row) % 7 == 0;
        };
        ParquetScan s = ParquetScan.scan(flat).select(ALL)
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.custom("mod7", "id"))
                .withCustomPredicate("mod7", evenScoreId).exactFilter();
        assertEquals(expectedIds(r -> r.id() % 7 == 0), ids(s));
    }

    @Test
    void customPredicateCombinesWithBuiltInLeaves() {
        CustomPredicate nameStartsWithA = (batch, row) -> {
            VarCharVector v = (VarCharVector) batch.getVector("name");
            return !v.isNull(row) && utf8(v, row).startsWith("a");
        };
        ParquetScan s = ParquetScan.scan(flat).select(ALL)
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.and(
                        com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("id", 100L),
                        com.github.gbenroscience.parser.ng.parquet.v1.Predicate.custom("a", "name")))
                .withCustomPredicate("a", nameStartsWithA).exactFilter();
        assertEquals(expectedIds(r -> r.id() < 100 && r.name() != null && r.name().startsWith("a")), ids(s));
    }

    @Test
    void reRegisteringAnIdReplacesTheEarlierImplementation() {
        ParquetScan s = ParquetScan.scan(flat).select("id")
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.custom("c", "id"))
                .withCustomPredicate("c", (b, r) -> false)
                .withCustomPredicate("c", (b, r) -> true).exactFilter();
        assertEquals(N, ids(s).size());
    }

    @Test
    void customPredicateExceptionsPropagateUncaught() {
        ParquetScan s = ParquetScan.scan(flat).select("id")
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.custom("boom", "id"))
                .withCustomPredicate("boom", (b, r) -> { throw new IllegalStateException("custom failure"); })
                .exactFilter();
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = s.open(a)) {
            RuntimeException e = assertThrows(RuntimeException.class, r::next);
            Throwable root = e;
            while (root.getCause() != null) root = root.getCause();
            assertEquals("custom failure", root.getMessage());
        }
    }

    @Test
    void unregisteredCustomIdFailsAtOpenNamingTheId() {
        ParquetScan s = ParquetScan.scan(flat).select("id")
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.custom("missing", "id")).exactFilter();
        try (BufferAllocator a = new RootAllocator()) {
            ParquetScanException e = assertThrows(ParquetScanException.class, () -> s.open(a));
            assertTrue(e.getMessage().contains("missing"), e.getMessage());
            assertEquals(0L, a.getAllocatedMemory(), "a failed open must not leak");
        }
    }

    @Test
    void customPredicateTouchingAnUnprojectedColumnFailsAtOpen() {
        ParquetScan s = ParquetScan.scan(flat).select("id")
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.custom("c", "name"))
                .withCustomPredicate("c", (b, r) -> true).exactFilter();
        try (BufferAllocator a = new RootAllocator()) {
            ParquetScanException e = assertThrows(ParquetScanException.class, () -> s.open(a));
            assertEquals("name", e.column());
        }
    }

    // ================================================================ validation at open

    private static ParquetScanException openFails(ParquetScan s) {
        try (BufferAllocator a = new RootAllocator()) {
            return assertThrows(ParquetScanException.class, () -> s.open(a).close());
        }
    }

    @Test
    void predicateColumnMustBeProjected() {
        ParquetScanException e = openFails(ParquetScan.scan(flat).select("id")
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("grp", 1)).exactFilter());
        assertEquals("grp", e.column());
    }

    @Test
    void unknownPredicateColumnFailsNamingIt() {
        ParquetScanException e = openFails(ParquetScan.scan(flat)
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("nope", 1)).exactFilter());
        assertEquals("nope", e.column());
    }

    @Test
    void literalTypesAreNeverCoercedBeyondTheDocumentedWidening() {
        for (com.github.gbenroscience.parser.ng.parquet.v1.Predicate bad : List.of(
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("grp", 1L),          // Long on INT32
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("grp", "1"),
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("id", 1.5d),
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("name", 1),
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("flag", 1),
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("day", 18_000),       // int on DATE
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("ratio", 1.0d),       // Double on FLOAT
                com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("score", Double.NaN))) {
            ParquetScanException e = openFails(ParquetScan.scan(flat).select(ALL).pushdown(bad).exactFilter());
            assertNotNull(e.column(), "should name the column for " + bad);
        }
    }

    @Test
    void booleanColumnsRejectRangeOperators() {
        openFails(ParquetScan.scan(flat).select(ALL)
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.gt("flag", true)).exactFilter());
    }

    @Test
    void likeAndRegexRequireAStringColumn() {
        openFails(ParquetScan.scan(flat).select(ALL).pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.like("id", "1%")).exactFilter());
        openFails(ParquetScan.scan(flat).select(ALL).pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.regex("grp", "1")).exactFilter());
    }

    @Test
    void columnToColumnRequiresIdenticalTypes() {
        ParquetScanException e = openFails(ParquetScan.scan(flat).select(ALL)
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.colLt("id", "score")).exactFilter());
        assertTrue(e.getMessage().contains("matching types"), e.getMessage());
    }

    // ================================================================ output shape

    @Test
    void exactBatchesAreDenseAndNeverEmpty() {
        try (BufferAllocator a = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(flat).select("id").batchSize(200)
                     .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.eq("id", 4_321L)).exactFilter().open(a)) {
            int batches = 0, rows = 0;
            while (r.next()) {
                batches++;
                assertTrue(r.root().getRowCount() > 0, "empty batches must be skipped, not surfaced");
                rows += r.root().getRowCount();
            }
            assertEquals(1, rows);
            assertEquals(1, batches);
        }
    }

    @Test
    void exactFilterReleasesAllMemoryOnClose() {
        try (BufferAllocator a = new RootAllocator()) {
            ParquetBatchReader r = ParquetScan.scan(flat).select(ALL).batchSize(300)
                    .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.ge("id", 100L)).exactFilter().open(a);
            assertTrue(r.next());
            r.close();
            assertEquals(0L, a.getAllocatedMemory());
        }
    }

    @Test
    void detachedExactBatchOutlivesTheReader() {
        try (BufferAllocator a = new RootAllocator()) {
            org.apache.arrow.vector.VectorSchemaRoot d;
            try (ParquetBatchReader r = ParquetScan.scan(flat).select("id")
                    .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("id", 10L)).exactFilter().open(a)) {
                assertTrue(r.next());
                d = r.detach();
            }
            try (org.apache.arrow.vector.VectorSchemaRoot owned = d) {
                assertEquals(10, owned.getRowCount());
                assertEquals(9L, ((BigIntVector) owned.getVector("id")).get(9));
            }
            assertEquals(0L, a.getAllocatedMemory());
        }
    }

    @Test
    void exactFilterHoldsWhenNoRowMatches() {
        assertEquals(List.of(), ids(ParquetScan.scan(flat).select("id")
                .pushdown(com.github.gbenroscience.parser.ng.parquet.v1.Predicate.lt("id", 0L)).exactFilter()));
    }
}