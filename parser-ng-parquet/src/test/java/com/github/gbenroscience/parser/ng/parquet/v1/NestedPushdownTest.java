package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.IntPredicate;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Predicates over leaves inside structs and lists: pruning must skip row groups (and never drop a match),
 * and {@code exactFilter()} must give "some element satisfies" semantics through lists, with null structs
 * and null/empty lists handled exactly.
 */
class NestedPushdownTest {

    private static final int N = 4_000;

    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message events {"
            + " required int64 id;"
            + " optional group user { required int32 tier; optional binary country (STRING); }"
            + " optional group items (LIST) { repeated group list {"
            + "   optional group element { required int64 price; required binary sku (STRING); } } }"
            + "}");

    @TempDir
    static Path dir;
    static Path file;

    // ---- the fixture, described once, used both to write and as the test oracle ----

    /** user is null every 13th row; country is null every 11th; tier rises with id so row groups have narrow ranges. */
    static boolean userNull(int i) { return i % 13 == 0; }
    static int tier(int i) { return i / 500; }
    static boolean countryNull(int i) { return i % 11 == 0; }
    static String country(int i) { return i % 3 == 0 ? "NG" : "GH"; }

    /** items: null every 50th row (i%50==49), empty when i%50==48, otherwise two elements with rising prices. */
    static List<Long> prices(int i) {
        if (i % 50 >= 48) return List.of();
        return List.of(i * 10L, i * 10L + 5L);
    }
    static List<String> skus(int i) {
        if (i % 50 >= 48) return List.of();
        return List.of("s" + (i % 7), "t" + (i % 5));
    }

    @BeforeAll
    static void write() throws IOException {
        file = dir.resolve("events.parquet");
        SimpleGroupFactory f = new SimpleGroupFactory(SCHEMA);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(SCHEMA).withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
                .withRowGroupSize(4 * 1024).build()) {
            for (int i = 0; i < N; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                if (!userNull(i)) {
                    Group u = g.addGroup("user");
                    u.append("tier", tier(i));
                    if (!countryNull(i)) u.append("country", country(i));
                }
                if (i % 50 != 49) {
                    Group items = g.addGroup("items");
                    List<Long> p = prices(i);
                    List<String> s = skus(i);
                    for (int k = 0; k < p.size(); k++) {
                        Group e = items.addGroup("list").addGroup("element");
                        e.append("price", p.get(k));
                        e.append("sku", s.get(k));
                    }
                }
                w.write(g);
            }
        }
    }

    // ---- helpers ----

    private static List<Long> expected(IntPredicate oracle) {
        List<Long> out = new ArrayList<>();
        for (int i = 0; i < N; i++) if (oracle.test(i)) out.add((long) i);
        return out;
    }

    private static List<Long> ids(ParquetScan scan) {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = scan.open(a)) {
            List<Long> out = new ArrayList<>();
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                BigIntVector id = (BigIntVector) b.getVector("id");
                for (int i = 0; i < b.getRowCount(); i++) out.add(id.get(i));
            }
            return out;
        }
    }

    private static ScanMetrics metricsAfterDraining(ParquetScan scan) {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = scan.withMetrics(true).open(a)) {
            while (r.next()) { /* drain */ }
            return r.metrics();
        }
    }

    private static void assertSuperset(List<Long> superset, List<Long> required, String what) {
        Set<Long> got = new HashSet<>(superset);
        for (Long id : required) assertTrue(got.contains(id), what + ": pruning dropped matching row " + id);
    }

    private static ParquetScan exact(Predicate p) {
        return ParquetScan.scan(file).select("id", "user", "items").pushdown(p).exactFilter();
    }

    // ================================================================ pruning

    @Test
    void fixtureSpansManyRowGroups() {
        assertTrue(ParquetFileInfo.read(file).rowGroups().size() >= 4);
    }

    @Test
    void structLeafSkipsRowGroupsAndKeepsEveryMatch() {
        ParquetScan s = ParquetScan.scan(file).select("id").pushdown(Predicate.gt("user.tier", 5));
        assertTrue(metricsAfterDraining(s).rowGroupsSkipped() > 0, "a struct-nested leaf should prune row groups");
        assertSuperset(ids(s), expected(i -> !userNull(i) && tier(i) > 5), "user.tier > 5");
    }

    @Test
    void listLeafSkipsRowGroupsAndKeepsEveryMatch() {
        ParquetScan s = ParquetScan.scan(file).select("id").pushdown(Predicate.gt("items.price", 30_000L));
        assertTrue(metricsAfterDraining(s).rowGroupsSkipped() > 0, "a leaf inside a list should prune row groups");
        assertSuperset(ids(s), expected(i -> prices(i).stream().anyMatch(p -> p > 30_000L)), "items.price > 30000");
    }

    @Test
    void listLeafInEqualityAndInListPrune() {
        ParquetScan eq = ParquetScan.scan(file).select("id").pushdown(Predicate.eq("items.price", 10_005L));
        assertTrue(metricsAfterDraining(eq).rowGroupsSkipped() > 0);
        assertSuperset(ids(eq), expected(i -> prices(i).contains(10_005L)), "items.price = 10005");

        ParquetScan in = ParquetScan.scan(file).select("id").pushdown(Predicate.in("items.price", 5L, 39_995L));
        assertTrue(metricsAfterDraining(in).rowGroupsSkipped() > 0);
        assertSuperset(ids(in), expected(i -> prices(i).contains(5L) || prices(i).contains(39_995L)), "items.price IN");
    }

    @Test
    void negationOverAListLeafNeverPrunesAwayEmptyOrNullLists() {
        // NOT(some element > 100) means NO element > 100: rows 0..9 and every null/empty-list row match.
        Predicate p = Predicate.not(Predicate.gt("items.price", 100L));
        List<Long> want = expected(i -> prices(i).stream().noneMatch(x -> x > 100L));
        assertSuperset(ids(ParquetScan.scan(file).select("id").pushdown(p)), want, "NOT(items.price > 100)");
        assertEquals(want, ids(exact(p)));
    }

    @Test
    void orMixingAListLeafWithAFlatLeafStaysSound() {
        Predicate p = Predicate.or(Predicate.gt("items.price", 39_000L), Predicate.lt("user.tier", 1));
        List<Long> want = expected(i -> prices(i).stream().anyMatch(x -> x > 39_000L) || (!userNull(i) && tier(i) < 1));
        assertSuperset(ids(ParquetScan.scan(file).select("id").pushdown(p)), want, "mixed OR");
        assertEquals(want, ids(exact(p)));
    }

    @Test
    void parallelScansPruneTheSameWay() {
        Predicate p = Predicate.gt("items.price", 30_000L);
        ScanMetrics one = metricsAfterDraining(ParquetScan.scan(file).select("id").pushdown(p));
        ScanMetrics three = metricsAfterDraining(ParquetScan.scan(file).select("id").pushdown(p).parallelism(3));
        assertEquals(one.rowGroupsAfterPruning(), three.rowGroupsAfterPruning());
        assertEquals(one.rowGroupsInFile(), three.rowGroupsInFile());
    }

    // ================================================================ exactFilter

    @Test
    void structLeafExactComparisonTreatsANullStructAsNull() {
        assertEquals(expected(i -> !userNull(i) && tier(i) >= 6), ids(exact(Predicate.ge("user.tier", 6))));
        assertEquals(expected(i -> !userNull(i) && !countryNull(i) && country(i).equals("NG")),
                ids(exact(Predicate.eq("user.country", "NG"))));
    }

    @Test
    void isNullOnAStructLeafIsTrueForNullParentAndNullLeaf() {
        assertEquals(expected(i -> userNull(i) || countryNull(i)), ids(exact(Predicate.isNull("user.country"))));
        assertEquals(expected(i -> !userNull(i) && !countryNull(i)), ids(exact(Predicate.isNotNull("user.country"))));
    }

    @Test
    void listLeafExactMeansSomeElementSatisfies() {
        assertEquals(expected(i -> prices(i).stream().anyMatch(p -> p > 39_990L)),
                ids(exact(Predicate.gt("items.price", 39_990L))));
        assertEquals(expected(i -> prices(i).contains(1_005L)), ids(exact(Predicate.eq("items.price", 1_005L))));
        assertEquals(expected(i -> skus(i).contains("s3")), ids(exact(Predicate.eq("items.sku", "s3"))));
        assertEquals(expected(i -> skus(i).stream().anyMatch(x -> x.startsWith("t4"))),
                ids(exact(Predicate.like("items.sku", "t4%"))));
    }

    @Test
    void conditionsOnOneListAreEachAnyNotJointlyAny() {
        // Two separate leaves under one AND: each is its own "some element", they need not be the same element.
        Predicate p = Predicate.and(Predicate.eq("items.sku", "s0"), Predicate.eq("items.sku", "t0"));
        assertEquals(expected(i -> skus(i).contains("s0") && skus(i).contains("t0")), ids(exact(p)));
    }

    @Test
    void isNullOverAListLeafMeansSomeElementIsNullAndEmptyListsHaveNone() {
        // No element here is ever null, so IS NULL matches nothing (a null or empty list has no element at all)
        // and IS NOT NULL matches every row that has at least one element.
        assertEquals(List.of(), ids(exact(Predicate.isNull("items.price"))));
        assertEquals(expected(i -> !prices(i).isEmpty()), ids(exact(Predicate.isNotNull("items.price"))));
    }

    @Test
    void physicalAndLogicalSpellingsOfAListPathAgree() {
        List<Long> a = ids(exact(Predicate.gt("items.price", 39_000L)));
        assertEquals(a, ids(exact(Predicate.gt("items.element.price", 39_000L))));
        assertEquals(a, ids(exact(Predicate.gt("items.list.element.price", 39_000L))));
        assertFalse(a.isEmpty());
    }

    @Test
    void parallelExactScanMatchesSequential() {
        Predicate p = Predicate.and(Predicate.gt("items.price", 20_000L), Predicate.ge("user.tier", 4));
        assertEquals(ids(exact(p)), ids(exact(p).parallelism(3)));
    }

    @Test
    void exactFilterStillRequiresTheTopLevelColumnToBeProjected() {
        ParquetScanException e = assertThrows(ParquetScanException.class, () -> {
            try (BufferAllocator a = new RootAllocator()) {
                ParquetScan.scan(file).select("id").pushdown(Predicate.gt("items.price", 1L)).exactFilter().open(a).close();
            }
        });
        assertEquals("items.price", e.column());
    }

    @Test
    void unknownNestedPathFailsNamingIt() {
        ParquetScanException e = assertThrows(ParquetScanException.class, () -> {
            try (BufferAllocator a = new RootAllocator()) {
                exact(Predicate.gt("items.nope", 1L)).open(a).close();
            }
        });
        assertEquals("items.nope", e.column());
    }
}
