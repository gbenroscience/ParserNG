package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import org.apache.parquet.filter2.predicate.FilterPredicate;
import org.apache.parquet.filter2.predicate.Operators;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.*;

/**
 * PredicateTranslator decides what may be pushed to parquet-java for pruning. The rule being tested:
 * translate exactly when the literal type is an exact/widening match, otherwise return null (no pruning), never coerce.
 */
class PredicateTranslatorTest {

    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message t {"
            + " required int64 id;"
            + " optional int32 grp;"
            + " optional binary name (STRING);"
            + " required double score;"
            + " required boolean flag;"
            + " optional float f;"
            + " optional int32 day (DATE);"
            + " optional int64 ts (TIMESTAMP(MILLIS,true));"
            + " optional group pt { required double x; }"
            + "}");

    private static FilterPredicate tr(Predicate p) {
        return PredicateTranslator.translate(p, SCHEMA);
    }

    private static void assertSameTranslation(Predicate expected, Predicate actual) {
        FilterPredicate e = tr(expected), a = tr(actual);
        assertNotNull(e, "reference predicate should translate");
        assertNotNull(a);
        assertEquals(e.toString(), a.toString());
    }

    // ------------------------------------------------------------ leaves

    @Test
    void nullPredicateTranslatesToNull() {
        assertNull(PredicateTranslator.translate(null, SCHEMA));
    }

    @Test
    void exactTypeLiteralsTranslate() {
        assertNotNull(tr(Predicate.eq("id", 5L)));
        assertNotNull(tr(Predicate.eq("grp", 5)));
        assertNotNull(tr(Predicate.eq("name", "x")));
        assertNotNull(tr(Predicate.eq("score", 1.5d)));
        assertNotNull(tr(Predicate.eq("f", 1.5f)));
        assertNotNull(tr(Predicate.eq("flag", true)));
    }

    @Test
    void everyOperatorTranslatesOnANumericColumn() {
        for (Predicate.Op op : Predicate.Op.values()) {
            assertNotNull(tr(new Predicate.Cmp("id", op, 10L)), op.name());
        }
    }

    @Test
    void wideningIsAllowedButNarrowingAndCrossTypeIsNot() {
        assertNotNull(tr(Predicate.gt("id", 5)), "Integer widens to an INT64 column");
        assertNotNull(tr(Predicate.gt("score", 5)), "Integer widens to a DOUBLE column");
        assertNotNull(tr(Predicate.gt("score", 5f)), "Float widens to a DOUBLE column");
        assertNull(tr(Predicate.gt("grp", 5L)), "Long must not be narrowed onto INT32");
        assertNull(tr(Predicate.gt("f", 1.5d)), "Double must not be narrowed onto FLOAT");
        assertNull(tr(Predicate.eq("id", "5")), "String on a numeric column is never coerced");
        assertNull(tr(Predicate.eq("name", 5)), "number on a string column is never coerced");
        assertNull(tr(Predicate.eq("flag", 1)));
    }

    @Test
    void dateColumnsTakeLocalDateOnly() {
        assertNotNull(tr(Predicate.eq("day", LocalDate.of(2020, 1, 1))));
        assertNull(tr(Predicate.eq("day", 18262)));
    }

    @Test
    void timestampColumnsAreNeverPushed() {
        assertNull(tr(Predicate.gt("ts", 1L)));
    }

    @Test
    void stringRangeIsNeverPushedButEqualityIs() {
        assertNull(tr(Predicate.gt("name", "a")));
        assertNull(tr(Predicate.lt("name", "a")));
        assertNotNull(tr(Predicate.eq("name", "a")));
        assertNotNull(tr(Predicate.ne("name", "a")));
    }

    @Test
    void booleanSupportsOnlyEqualityOperators() {
        assertNotNull(tr(Predicate.eq("flag", true)));
        assertNotNull(tr(Predicate.ne("flag", true)));
        assertNull(tr(Predicate.gt("flag", true)));
    }

    @Test
    void nanLiteralsAreNeverPushed() {
        assertNull(tr(Predicate.gt("score", Double.NaN)));
        assertNull(tr(Predicate.gt("f", Float.NaN)));
    }

    @Test
    void unknownNestedAndDottedColumnsAreNotPushed() {
        assertNull(tr(Predicate.eq("nope", 1L)));
        assertNull(tr(Predicate.eq("pt", 1.0d)), "a group column has no statistics to prune on");
        assertNull(tr(Predicate.eq("pt.x", 1.0d)), "dotted paths are ambiguous");
    }

    @Test
    void inListTranslatesOnlyWhenEveryElementDoes() {
        assertNotNull(tr(Predicate.in("grp", 1, 2, 3)));
        assertNotNull(tr(Predicate.in("name", "a", "b")));
        assertNull(tr(Predicate.in("grp", 1, "two")));
        assertNull(tr(Predicate.in("grp", 1, 2L)));
        assertNull(tr(Predicate.in("name", "a", 2)));
    }

    @Test
    void isNullTranslatesOnPrimitiveColumns() {
        assertNotNull(tr(Predicate.isNull("grp")));
        assertNotNull(tr(Predicate.isNotNull("grp")));
        assertNotNull(tr(Predicate.isNull("name")));
        assertNull(tr(Predicate.isNull("pt")));
        assertNull(tr(Predicate.isNull("nope")));
    }

    // ------------------------------------------------------------ never-pushed leaves

    @Test
    void colCmpLikeRegexAndCustomNeverPush() {
        assertNull(tr(Predicate.colLt("id", "score")));
        assertNull(tr(Predicate.like("name", "a%")));
        assertNull(tr(Predicate.regex("name", "a.*")));
        assertNull(tr(Predicate.custom("rule", "id")));
    }

    // ------------------------------------------------------------ And / Or

    @Test
    void andOfTwoPushableLeavesIsAnAnd() {
        FilterPredicate fp = tr(Predicate.and(Predicate.gt("id", 1L), Predicate.lt("id", 9L)));
        assertInstanceOf(Operators.And.class, fp);
    }

    @Test
    void andDropsAnUntranslatableSideBecauseAnySubsetOfConjunctsIsSound() {
        Predicate keep = Predicate.gt("id", 1L);
        assertSameTranslation(keep, Predicate.and(keep, Predicate.like("name", "a%")));
        assertSameTranslation(keep, Predicate.and(Predicate.like("name", "a%"), keep));
    }

    @Test
    void andOfTwoUntranslatableSidesIsNull() {
        assertNull(tr(Predicate.and(Predicate.like("name", "a%"), Predicate.custom("c"))));
    }

    @Test
    void orOfTwoPushableLeavesIsAnOr() {
        assertInstanceOf(Operators.Or.class, tr(Predicate.or(Predicate.eq("id", 1L), Predicate.eq("id", 2L))));
    }

    @Test
    void orWithAnUntranslatableSideIsDroppedEntirelyBecauseThatSideCouldMatchAnything() {
        assertNull(tr(Predicate.or(Predicate.eq("id", 1L), Predicate.like("name", "a%"))));
        assertNull(tr(Predicate.or(Predicate.like("name", "a%"), Predicate.eq("id", 1L))));
    }

    // ------------------------------------------------------------ Not (negation-normal form)

    @Test
    void notNeverProducesAParquetNotNode() {
        FilterPredicate fp = tr(Predicate.not(Predicate.gt("id", 5L)));
        assertNotNull(fp);
        assertFalse(fp instanceof Operators.Not);
    }

    @Test
    void notFlipsEveryComparisonOperator() {
        assertSameTranslation(Predicate.ne("id", 5L), Predicate.not(Predicate.eq("id", 5L)));
        assertSameTranslation(Predicate.eq("id", 5L), Predicate.not(Predicate.ne("id", 5L)));
        assertSameTranslation(Predicate.ge("id", 5L), Predicate.not(Predicate.lt("id", 5L)));
        assertSameTranslation(Predicate.gt("id", 5L), Predicate.not(Predicate.le("id", 5L)));
        assertSameTranslation(Predicate.le("id", 5L), Predicate.not(Predicate.gt("id", 5L)));
        assertSameTranslation(Predicate.lt("id", 5L), Predicate.not(Predicate.ge("id", 5L)));
    }

    @Test
    void notFlipsIsNull() {
        assertSameTranslation(Predicate.isNotNull("grp"), Predicate.not(Predicate.isNull("grp")));
        assertSameTranslation(Predicate.isNull("grp"), Predicate.not(Predicate.isNotNull("grp")));
    }

    @Test
    void doubleNegationCancels() {
        assertSameTranslation(Predicate.eq("id", 5L), Predicate.not(Predicate.not(Predicate.eq("id", 5L))));
    }

    @Test
    void notAppliesDeMorganToAndAndOr() {
        Predicate a = Predicate.eq("id", 1L), b = Predicate.gt("grp", 3);
        assertSameTranslation(
                Predicate.or(Predicate.ne("id", 1L), Predicate.le("grp", 3)),
                Predicate.not(Predicate.and(a, b)));
        assertSameTranslation(
                Predicate.and(Predicate.ne("id", 1L), Predicate.le("grp", 3)),
                Predicate.not(Predicate.or(a, b)));
    }

    @Test
    void notInBecomesAConjunctionOfNotEquals() {
        FilterPredicate fp = tr(Predicate.not(Predicate.in("grp", 1, 2)));
        assertNotNull(fp);
        assertSameTranslation(Predicate.and(Predicate.ne("grp", 1), Predicate.ne("grp", 2)),
                Predicate.not(Predicate.in("grp", 1, 2)));
    }

    @Test
    void notAroundANonPushableLeafStaysNonPushable() {
        assertNull(tr(Predicate.not(Predicate.like("name", "a%"))));
        assertNull(tr(Predicate.not(Predicate.colEq("id", "score"))));
        assertNull(tr(Predicate.not(Predicate.regex("name", "a"))));
        assertNull(tr(Predicate.not(Predicate.custom("c"))));
    }
}