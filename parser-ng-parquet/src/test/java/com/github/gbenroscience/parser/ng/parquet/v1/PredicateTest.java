package com.github.gbenroscience.parser.ng.parquet.v1;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/** Pure-Java tests of the {@link Predicate} model: construction rules, equality and LIKE compilation. */
class PredicateTest {

    // ---------------------------------------------------------------- Cmp

    @Test
    void cmpFactoriesBuildTheMatchingOperator() {
        assertEquals(new Predicate.Cmp("a", Predicate.Op.EQ, 1), Predicate.eq("a", 1));
        assertEquals(new Predicate.Cmp("a", Predicate.Op.NE, 1), Predicate.ne("a", 1));
        assertEquals(new Predicate.Cmp("a", Predicate.Op.LT, 1), Predicate.lt("a", 1));
        assertEquals(new Predicate.Cmp("a", Predicate.Op.LE, 1), Predicate.le("a", 1));
        assertEquals(new Predicate.Cmp("a", Predicate.Op.GT, 1), Predicate.gt("a", 1));
        assertEquals(new Predicate.Cmp("a", Predicate.Op.GE, 1), Predicate.ge("a", 1));
    }

    @Test
    void cmpRejectsNullOrEmptyColumnNullOpAndNullValue() {
        assertThrows(IllegalArgumentException.class, () -> Predicate.eq(null, 1));
        assertThrows(IllegalArgumentException.class, () -> Predicate.eq("", 1));
        assertThrows(IllegalArgumentException.class, () -> new Predicate.Cmp("a", null, 1));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Predicate.eq("a", null));
        assertTrue(e.getMessage().contains("isNull"), "message should point callers at isNull()/isNotNull()");
    }

    // ---------------------------------------------------------------- In

    @Test
    void inCopiesItsValuesDefensively() {
        List<Object> mutable = new ArrayList<>(List.of(1, 2, 3));
        Predicate.In in = new Predicate.In("a", mutable);
        mutable.add(4);
        assertEquals(List.of(1, 2, 3), in.values());
        assertThrows(UnsupportedOperationException.class, () -> {
        List l = (List) in.values();
           l.add(9); 
        });
    }

    @Test
    void inRejectsEmptyAndNullValues() {
        assertThrows(IllegalArgumentException.class, () -> Predicate.in("a"));
        assertThrows(IllegalArgumentException.class, () -> new Predicate.In("a", null));
        assertThrows(IllegalArgumentException.class, () -> new Predicate.In("a", List.of()));
        assertThrows(NullPointerException.class, () -> Predicate.in("a", 1, null),
                "null list elements are rejected by List.copyOf");
    }

    @Test
    void inVarargsFactoryKeepsOrder() {
        assertEquals(List.of("x", "y", "z"), ((Predicate.In) Predicate.in("a", "x", "y", "z")).values());
    }

    // ---------------------------------------------------------------- IsNull

    @Test
    void isNullAndIsNotNullSetTheNegatedFlag() {
        assertFalse(((Predicate.IsNull) Predicate.isNull("a")).negated());
        assertTrue(((Predicate.IsNull) Predicate.isNotNull("a")).negated());
        assertThrows(IllegalArgumentException.class, () -> Predicate.isNull(null));
    }

    // ---------------------------------------------------------------- And / Or / Not

    @Test
    void booleanCombinatorsRejectNullOperands() {
        Predicate p = Predicate.eq("a", 1);
        assertThrows(IllegalArgumentException.class, () -> Predicate.and(null, p));
        assertThrows(IllegalArgumentException.class, () -> Predicate.and(p, null));
        assertThrows(IllegalArgumentException.class, () -> Predicate.or(null, p));
        assertThrows(IllegalArgumentException.class, () -> Predicate.or(p, null));
        assertThrows(IllegalArgumentException.class, () -> Predicate.not(null));
    }

    @Test
    void booleanCombinatorsKeepOperandsInOrder() {
        Predicate l = Predicate.eq("a", 1), r = Predicate.eq("b", 2);
        Predicate.And and = (Predicate.And) Predicate.and(l, r);
        assertSame(l, and.left());
        assertSame(r, and.right());
        Predicate.Or or = (Predicate.Or) Predicate.or(l, r);
        assertSame(l, or.left());
        assertSame(r, or.right());
        assertSame(l, ((Predicate.Not) Predicate.not(l)).inner());
    }

    // ---------------------------------------------------------------- ColCmp

    @Test
    void colCmpFactoriesAndValidation() {
        assertEquals(new Predicate.ColCmp("a", Predicate.Op.LT, "b"), Predicate.colLt("a", "b"));
        assertEquals(new Predicate.ColCmp("a", Predicate.Op.LE, "b"), Predicate.colLe("a", "b"));
        assertEquals(new Predicate.ColCmp("a", Predicate.Op.GT, "b"), Predicate.colGt("a", "b"));
        assertEquals(new Predicate.ColCmp("a", Predicate.Op.GE, "b"), Predicate.colGe("a", "b"));
        assertEquals(new Predicate.ColCmp("a", Predicate.Op.EQ, "b"), Predicate.colEq("a", "b"));
        assertEquals(new Predicate.ColCmp("a", Predicate.Op.NE, "b"), Predicate.colNe("a", "b"));
        assertThrows(IllegalArgumentException.class, () -> Predicate.colEq(null, "b"));
        assertThrows(IllegalArgumentException.class, () -> Predicate.colEq("a", ""));
        assertThrows(IllegalArgumentException.class, () -> new Predicate.ColCmp("a", null, "b"));
    }

    // ---------------------------------------------------------------- LIKE

    private static boolean likeMatches(Predicate p, String input) {
        return ((Predicate.Like) p).compiled().matcher(input).matches();
    }

    @Test
    void likePercentMatchesAnySequenceIncludingEmpty() {
        Predicate p = Predicate.like("c", "ab%");
        assertTrue(likeMatches(p, "ab"));
        assertTrue(likeMatches(p, "abcdef"));
        assertFalse(likeMatches(p, "a"));
        assertFalse(likeMatches(p, "xab"));
    }

    @Test
    void likeUnderscoreMatchesExactlyOneCharacter() {
        Predicate p = Predicate.like("c", "a_c");
        assertTrue(likeMatches(p, "abc"));
        assertFalse(likeMatches(p, "ac"));
        assertFalse(likeMatches(p, "abbc"));
    }

    @Test
    void likeTreatsRegexMetacharactersLiterally() {
        Predicate p = Predicate.like("c", "a.b(c)[d]+");
        assertTrue(likeMatches(p, "a.b(c)[d]+"));
        assertFalse(likeMatches(p, "axb(c)[d]+"));
    }

    @Test
    void likeDefaultEscapeIsBackslash() {
        Predicate p = Predicate.like("c", "100\\%");
        assertTrue(likeMatches(p, "100%"));
        assertFalse(likeMatches(p, "1000"));
        Predicate u = Predicate.like("c", "a\\_b");
        assertTrue(likeMatches(u, "a_b"));
        assertFalse(likeMatches(u, "axb"));
    }

    @Test
    void likeEscapeCanEscapeItself() {
        Predicate p = Predicate.like("c", "a\\\\b");
        assertTrue(likeMatches(p, "a\\b"));
    }

    @Test
    void likeCustomEscapeCharacter() {
        Predicate p = Predicate.like("c", "a#%b", false, '#');
        assertTrue(likeMatches(p, "a%b"));
        assertFalse(likeMatches(p, "aXXb"));
    }

    @Test
    void likeWithNullEscapeTreatsBackslashLiterally() {
        Predicate p = Predicate.like("c", "a\\b", false, null);
        assertTrue(likeMatches(p, "a\\b"));
    }

    @Test
    void likeDanglingOrInvalidEscapeFailsWhenBuilt() {
        assertThrows(IllegalArgumentException.class, () -> Predicate.like("c", "abc\\"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Predicate.like("c", "a\\b"));
        assertTrue(e.getMessage().contains("escape"));
    }

    @Test
    void ilikeIsCaseInsensitive() {
        Predicate p = Predicate.ilike("c", "hello%");
        assertTrue(likeMatches(p, "HELLO world"));
        assertTrue(likeMatches(p, "Hello"));
        assertFalse(likeMatches(Predicate.like("c", "hello%"), "HELLO world"));
    }

    @Test
    void likeValidatesColumnAndPattern() {
        assertThrows(IllegalArgumentException.class, () -> Predicate.like(null, "a"));
        assertThrows(IllegalArgumentException.class, () -> Predicate.like("c", null));
    }

    @Test
    void likeEqualityIgnoresTheCompiledPatternIdentity() {
        Predicate a = Predicate.like("c", "a%");
        Predicate b = Predicate.like("c", "a%");
        assertNotSame(((Predicate.Like) a).compiled(), ((Predicate.Like) b).compiled());
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, Predicate.like("c", "b%"));
        assertNotEquals(a, Predicate.ilike("c", "a%"));
        assertNotEquals(a, Predicate.like("d", "a%"));
        assertNotEquals(a, Predicate.like("c", "a%", false, '#'));
    }

    // ---------------------------------------------------------------- Regex

    @Test
    void regexFactoryCompilesAndRejectsBadSyntax() {
        Predicate.Regex r = (Predicate.Regex) Predicate.regex("c", "^a+$");
        assertTrue(r.pattern().matcher("aaa").find());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Predicate.regex("c", "(unclosed"));
        assertTrue(e.getMessage().contains("'c'"), "message should name the column");
        assertNotNull(e.getCause());
    }

    @Test
    void regexEqualityIsBySourceAndFlags() {
        Predicate a = Predicate.regex("c", "a+");
        Predicate b = Predicate.regex("c", "a+");
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, Predicate.regex("c", "b+"));
        assertNotEquals(a, Predicate.regex("d", "a+"));
        assertNotEquals(a, Predicate.regex("c", Pattern.compile("a+", Pattern.CASE_INSENSITIVE)));
    }

    @Test
    void regexRejectsNullPatternAndColumn() {
        assertThrows(IllegalArgumentException.class, () -> Predicate.regex("c", (Pattern) null));
        assertThrows(IllegalArgumentException.class, () -> Predicate.regex(null, "a"));
    }

    // ---------------------------------------------------------------- Custom

    @Test
    void customCarriesIdAndTouchedColumns() {
        Predicate.Custom c = (Predicate.Custom) Predicate.custom("rule", "a", "b");
        assertEquals("rule", c.id());
        assertEquals(Set.of("a", "b"), c.touchedColumns());
    }

    @Test
    void customWithNoColumnsHasEmptyTouchedSet() {
        assertEquals(Set.of(), ((Predicate.Custom) Predicate.custom("rule")).touchedColumns());
        assertEquals(Set.of(), new Predicate.Custom("rule", null).touchedColumns());
    }

    @Test
    void customRequiresAnId() {
        assertThrows(IllegalArgumentException.class, () -> Predicate.custom(null));
        assertThrows(IllegalArgumentException.class, () -> Predicate.custom(""));
    }

    @Test
    void customTouchedColumnsAreCopiedDefensively() {
        Set<String> cols = new HashSet<>(Arrays.asList("a"));
        Predicate.Custom c = new Predicate.Custom("rule", cols);
        cols.add("b");
        assertEquals(Set.of("a"), c.touchedColumns());
        assertThrows(UnsupportedOperationException.class, () -> c.touchedColumns().add("z"));
    }

    // ---------------------------------------------------------------- records

    @Test
    void structuralEqualityHoldsForNestedTrees() {
        Predicate t1 = Predicate.and(Predicate.gt("a", 1), Predicate.or(Predicate.isNull("b"), Predicate.in("c", 1, 2)));
        Predicate t2 = Predicate.and(Predicate.gt("a", 1), Predicate.or(Predicate.isNull("b"), Predicate.in("c", 1, 2)));
        assertEquals(t1, t2);
        assertEquals(t1.hashCode(), t2.hashCode());
        assertNotEquals(t1, Predicate.and(Predicate.gt("a", 2), Predicate.or(Predicate.isNull("b"), Predicate.in("c", 1, 2))));
    }
}