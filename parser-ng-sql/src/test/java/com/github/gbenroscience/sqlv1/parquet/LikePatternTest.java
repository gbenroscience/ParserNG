package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

/** The byte-level LIKE matcher: every shape, UTF-8 safety of {@code _}, escapes, ASCII-only ILIKE. */
class LikePatternTest {

    private static boolean like(String pattern, String value) {
        return like(pattern, -1, false, value);
    }

    private static boolean like(String pattern, int esc, boolean ci, String value) {
        byte[] b = value.getBytes(StandardCharsets.UTF_8);
        // Embed in a larger buffer to prove the matcher honours [s, e).
        byte[] padded = new byte[b.length + 4];
        System.arraycopy(b, 0, padded, 2, b.length);
        return LikePattern.compile(pattern, esc, ci).matches(padded, 2, 2 + b.length);
    }

    @Test
    void equalsPrefixSuffixContains() {
        assertTrue(like("abc", "abc"));
        assertFalse(like("abc", "abcd"));
        assertTrue(like("abc%", "abcdef"));
        assertTrue(like("abc%", "abc"));
        assertFalse(like("abc%", "ab"));
        assertTrue(like("%def", "abcdef"));
        assertFalse(like("%def", "abcde"));
        assertTrue(like("%cd%", "abcdef"));
        assertFalse(like("%xy%", "abcdef"));
        assertTrue(like("%", ""));
        assertTrue(like("%%", "anything"));
        assertTrue(like("", ""));
        assertFalse(like("", "a"));
    }

    @Test
    void underscoreMatchesExactlyOneCharacter() {
        assertTrue(like("a_c", "abc"));
        assertFalse(like("a_c", "ac"));
        assertFalse(like("a_c", "abbc"));
        assertTrue(like("___", "xyz"));
    }

    @Test
    void underscoreConsumesAWholeUtf8Sequence() {
        assertTrue(like("a_c", "a\u00e9c"), "two-byte character");
        assertTrue(like("a_c", "a\u20acc"), "three-byte character");
        assertTrue(like("a_c", "a\uD83D\uDE00c"), "four-byte character (surrogate pair)");
        assertFalse(like("a__c", "a\u00e9c"), "one character must not count as two");
    }

    @Test
    void generalPatternsWithBacktracking() {
        assertTrue(like("a%b%c", "aXXbYYc"));
        assertTrue(like("a%b%c", "abc"));
        assertFalse(like("a%b%c", "aXXbYY"));
        assertTrue(like("%a_c%", "zzabczz"));
        assertTrue(like("a%aab", "aaaab"), "must retry after a partial match");
        assertFalse(like("a%aab", "aaaa"));
        assertTrue(like("_%_", "ab"));
        assertFalse(like("_%_", "a"));
    }

    @Test
    void escapeMakesWildcardsLiteral() {
        assertTrue(like("100!%", '!', false, "100%"));
        assertFalse(like("100!%", '!', false, "1000"));
        assertTrue(like("a!_b", '!', false, "a_b"));
        assertFalse(like("a!_b", '!', false, "axb"));
        assertTrue(like("!!x", '!', false, "!x"), "escaped escape");
    }

    @Test
    void withoutEscapeBackslashIsAnOrdinaryCharacter() {
        assertTrue(like("a\\b", "a\\b"));
    }

    @Test
    void danglingEscapeIsRejected() {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> LikePattern.compile("abc!", '!', false));
        assertTrue(e.getMessage().contains("escape"));
    }

    @Test
    void ilikeFoldsAsciiOnly() {
        assertTrue(like("HeLLo%", -1, true, "hello world"));
        assertTrue(like("%WORLD", -1, true, "hello world"));
        assertTrue(like("h_llo", -1, true, "HELLO"));
        assertFalse(like("\u00c9", -1, true, "\u00e9"), "non-ASCII letters are not folded (documented)");
        assertFalse(like("hello", -1, false, "HELLO"));
    }
}
