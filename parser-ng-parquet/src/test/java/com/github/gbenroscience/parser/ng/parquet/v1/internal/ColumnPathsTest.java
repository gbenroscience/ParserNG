package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import org.apache.parquet.filter2.predicate.FilterPredicate;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.github.gbenroscience.parser.ng.parquet.v1.internal.ColumnPaths.Step;
import static com.github.gbenroscience.parser.ng.parquet.v1.internal.ColumnPaths.StepKind.FIELD;
import static com.github.gbenroscience.parser.ng.parquet.v1.internal.ColumnPaths.StepKind.LIST;
import static org.junit.jupiter.api.Assertions.*;

/** Path resolution over pure schemas (no files): what a dotted name means, and which route reaches the leaf. */
class ColumnPathsTest {

    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message t {"
            + " required int64 id;"
            + " optional group user { optional group address { optional binary zip (STRING); } required int32 tier; }"
            + " optional group items (LIST) { repeated group list {"
            + "   optional group element { required int64 price; optional binary sku (STRING); } } }"
            + " optional group tags (LIST) { repeated group list { optional binary element (STRING); } }"
            + " optional group attrs (MAP) { repeated group key_value { required binary key (STRING); optional int64 value; } }"
            + " repeated int32 nums;"
            + " optional group legacy (LIST) { repeated int32 array; }"
            + "}");

    private static ColumnPaths.Resolved ok(String col) {
        ColumnPaths.Resolved r = ColumnPaths.resolve(SCHEMA, col);
        assertNotNull(r, col + " should resolve");
        return r;
    }

    private static Step field(String n) { return new Step(FIELD, n); }
    private static final Step LST = new Step(LIST, null);

    @Test
    void flatColumn() {
        ColumnPaths.Resolved r = ok("id");
        assertTrue(r.flat());
        assertFalse(r.repeated());
        assertArrayEquals(new String[] {"id"}, r.physicalPath());
    }

    @Test
    void structPathHasOneFieldHopPerLevelAndIsNotRepeated() {
        ColumnPaths.Resolved r = ok("user.address.zip");
        assertEquals("user", r.topLevel());
        assertEquals("user.address.zip", r.physicalDotted());
        assertEquals(List.of(field("address"), field("zip")), r.steps());
        assertFalse(r.repeated());
        assertEquals(List.of(field("tier")), ok("user.tier").steps());
    }

    @Test
    void listOfStructsAcceptsPhysicalLogicalAndImplicitElementSpellings() {
        for (String spelling : List.of("items.list.element.price", "items.element.price", "items.price")) {
            ColumnPaths.Resolved r = ok(spelling);
            assertEquals("items.list.element.price", r.physicalDotted(), spelling);
            assertEquals(List.of(LST, field("price")), r.steps(), spelling);
            assertTrue(r.repeated(), spelling);
            assertEquals(1, r.maxRep(), spelling);
        }
    }

    @Test
    void listOfPrimitives() {
        for (String spelling : List.of("tags.list.element", "tags.element")) {
            ColumnPaths.Resolved r = ok(spelling);
            assertEquals("tags.list.element", r.physicalDotted());
            assertEquals(List.of(LST), r.steps());
        }
    }

    @Test
    void mapKeyAndValue() {
        ColumnPaths.Resolved v = ok("attrs.value");
        assertEquals("attrs.key_value.value", v.physicalDotted());
        assertEquals(List.of(LST, field("value")), v.steps());
        assertEquals("attrs.key_value.key", ok("attrs.key_value.key").physicalDotted());
    }

    @Test
    void bareRepeatedAndLegacyTwoLevelLists() {
        ColumnPaths.Resolved nums = ok("nums");
        assertEquals(List.of(LST), nums.steps());
        assertTrue(nums.repeated());
        ColumnPaths.Resolved legacy = ok("legacy.array");
        assertEquals(List.of(LST), legacy.steps());
        assertEquals("legacy.array", legacy.physicalDotted());
    }

    @Test
    void namesThatAreNotExactlyOnePrimitiveLeafDoNotResolve() {
        for (String bad : List.of("nope", "user", "user.address", "user.nope", "user.address.zip.deeper",
                "items", "items.list", "items.list.element", "items.nope", "tags", "attrs", "attrs.key_value",
                "", "user.", ".user", "user..tier")) {
            assertNull(ColumnPaths.resolve(SCHEMA, bad), "'" + bad + "' must not resolve");
        }
        assertNull(ColumnPaths.resolve(SCHEMA, null));
        assertNull(ColumnPaths.resolve(null, "id"));
    }

    // ------------------------------------------------------------------ what pruning may push to parquet-java

    private static FilterPredicate tr(Predicate p) { return PredicateTranslator.translate(p, SCHEMA); }

    @Test
    void structLeavesArePushedUnderTheirPhysicalDottedName() {
        FilterPredicate fp = tr(Predicate.eq("user.address.zip", "10001"));
        assertNotNull(fp);
        assertTrue(fp.toString().contains("user.address.zip"), fp.toString());
        assertNotNull(tr(Predicate.gt("user.tier", 3)));
        assertNull(tr(Predicate.isNull("user.address.zip")));
    }

    @Test
    void leavesUnderListsAndMapsAreNeverHandedToParquetJava() {
        assertNull(tr(Predicate.gt("items.price", 5L)));
        assertNull(tr(Predicate.eq("tags.element", "x")));
        assertNull(tr(Predicate.eq("attrs.value", 1L)));
        assertNull(tr(Predicate.gt("nums", 1)));
        assertNull(tr(Predicate.in("items.price", 1L, 2L)));
        assertNull(tr(Predicate.isNull("items.price")));
    }

    @Test
    void negationNeverTurnsAListLeafIntoSomethingPushable() {
        assertNull(tr(Predicate.not(Predicate.gt("items.price", 5L))));
        assertNull(tr(Predicate.not(Predicate.and(Predicate.gt("items.price", 5L), Predicate.lt("id", 3L)))),
                "NOT(A AND B) = NOT A OR NOT B: one side is unpushable so the whole OR is dropped");
    }

    @Test
    void anAndKeepsThePushableSideAndAnOrDropsEverything() {
        FilterPredicate flatOnly = tr(Predicate.gt("id", 3L));
        assertNotNull(flatOnly);
        assertEquals(flatOnly.toString(), tr(Predicate.and(Predicate.gt("items.price", 5L), Predicate.gt("id", 3L))).toString());
        assertNull(tr(Predicate.or(Predicate.gt("items.price", 5L), Predicate.gt("id", 3L))));
        // NOT(A OR B) = NOT A AND NOT B: the pushable side survives
        assertNotNull(tr(Predicate.not(Predicate.or(Predicate.gt("items.price", 5L), Predicate.gt("id", 3L)))));
    }
}
