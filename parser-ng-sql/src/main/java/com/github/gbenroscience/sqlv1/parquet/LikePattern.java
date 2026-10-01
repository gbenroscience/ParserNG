package com.github.gbenroscience.sqlv1.parquet;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * A compiled SQL {@code LIKE} pattern that matches UTF-8 bytes directly: no {@code String} is built
 * per row.
 *
 * <p>{@code %} matches any run of characters (including none), {@code _} exactly one character (a
 * whole UTF-8 sequence, never half of one). An optional escape character makes the next pattern
 * character literal. Common shapes take fast paths: {@code abc} (equality), {@code abc%} (prefix),
 * {@code %abc} (suffix), {@code %abc%} (substring); anything else uses an iterative matcher with
 * bounded backtracking (no recursion, no allocation).
 *
 * <p><b>Case-insensitive ({@code ILIKE}) folds ASCII letters only.</b> {@code 'É'} does not match
 * {@code 'é'}. This is deliberate: full Unicode case folding cannot be done bytewise, and silently
 * approximating it would be worse than a documented limit.
 */
final class LikePattern {

    private static final int LIT = 0, ONE = 1, MANY = 2;

    private enum Shape { EQUALS, PREFIX, SUFFIX, CONTAINS, GENERAL }

    private final int[] kind;
    private final byte[] bytes;   // valid where kind == LIT
    private final boolean ci;
    private final Shape shape;
    private final byte[] needle;  // literal run for the fast shapes

    private LikePattern(int[] kind, byte[] bytes, boolean ci) {
        this.kind = kind;
        this.bytes = bytes;
        this.ci = ci;
        int n = kind.length;
        boolean allLit = true;
        for (int k : kind) allLit &= k == LIT;
        Shape s = Shape.GENERAL;
        int from = 0, to = n;
        if (allLit) {
            s = Shape.EQUALS;
        } else if (n >= 1 && kind[n - 1] == MANY && litRun(0, n - 1)) {
            s = Shape.PREFIX; to = n - 1;
        } else if (n >= 1 && kind[0] == MANY && litRun(1, n)) {
            s = Shape.SUFFIX; from = 1;
        } else if (n >= 2 && kind[0] == MANY && kind[n - 1] == MANY && litRun(1, n - 1)) {
            s = Shape.CONTAINS; from = 1; to = n - 1;
        }
        this.shape = s;
        this.needle = s == Shape.GENERAL ? null : java.util.Arrays.copyOfRange(bytes, from, to);
    }

    private boolean litRun(int from, int to) {
        for (int i = from; i < to; i++) if (kind[i] != LIT) return false;
        return true;
    }

    /**
     * @param pattern the pattern text
     * @param escape the escape code point, or {@code -1} for none
     * @param caseInsensitive fold ASCII letters
     * @throws ParquetSqlException if the pattern ends with a dangling escape
     */
    static LikePattern compile(String pattern, int escape, boolean caseInsensitive) {
        List<Integer> kinds = new ArrayList<>();
        List<Byte> litBytes = new ArrayList<>();
        int i = 0;
        while (i < pattern.length()) {
            int cp = pattern.codePointAt(i);
            i += Character.charCount(cp);
            boolean literal = false;
            if (escape >= 0 && cp == escape) {
                if (i >= pattern.length()) {
                    throw new ParquetSqlException("LIKE pattern '" + pattern + "' ends with the escape character");
                }
                cp = pattern.codePointAt(i);
                i += Character.charCount(cp);
                literal = true;
            }
            if (!literal && cp == '%') {
                if (kinds.isEmpty() || kinds.get(kinds.size() - 1) != MANY) { kinds.add(MANY); litBytes.add((byte) 0); }
            } else if (!literal && cp == '_') {
                kinds.add(ONE);
                litBytes.add((byte) 0);
            } else {
                for (byte b : new String(Character.toChars(cp)).getBytes(StandardCharsets.UTF_8)) {
                    kinds.add(LIT);
                    litBytes.add(caseInsensitive ? fold(b) : b);
                }
            }
        }
        int[] k = new int[kinds.size()];
        byte[] b = new byte[k.length];
        for (int j = 0; j < k.length; j++) { k[j] = kinds.get(j); b[j] = litBytes.get(j); }
        return new LikePattern(k, b, caseInsensitive);
    }

    private static byte fold(byte b) {
        return (b >= 'A' && b <= 'Z') ? (byte) (b + 32) : b;
    }

    private byte at(byte[] d, int i) {
        byte b = d[i];
        return ci ? fold(b) : b;
    }

    /** Matches {@code d[s, e)}. */
    boolean matches(byte[] d, int s, int e) {
        int len = e - s;
        switch (shape) {
            case EQUALS:
                return len == needle.length && regionEquals(d, s);
            case PREFIX:
                return len >= needle.length && regionEquals(d, s);
            case SUFFIX:
                return len >= needle.length && regionEquals(d, e - needle.length);
            case CONTAINS:
                for (int p = s; p + needle.length <= e; p++) if (regionEquals(d, p)) return true;
                return needle.length == 0;
            default:
                return general(d, s, e);
        }
    }

    private boolean regionEquals(byte[] d, int at) {
        for (int k = 0; k < needle.length; k++) if (at(d, at + k) != needle[k]) return false;
        return true;
    }

    private static int charLen(byte b) {
        if ((b & 0x80) == 0) return 1;
        if ((b & 0xE0) == 0xC0) return 2;
        if ((b & 0xF0) == 0xE0) return 3;
        if ((b & 0xF8) == 0xF0) return 4;
        return 1; // stray continuation byte: consume it alone
    }

    /** Iterative wildcard matcher; on a mismatch after a {@code %} it retries one character further on. */
    private boolean general(byte[] d, int s, int e) {
        int si = s, pi = 0, starPi = -1, starSi = -1;
        int m = kind.length;
        while (si < e) {
            if (pi < m && kind[pi] == LIT && at(d, si) == bytes[pi]) {
                si++; pi++;
            } else if (pi < m && kind[pi] == ONE) {
                si = Math.min(e, si + charLen(d[si])); pi++;
            } else if (pi < m && kind[pi] == MANY) {
                starPi = pi++; starSi = si;
            } else if (starPi >= 0) {
                pi = starPi + 1;
                starSi = Math.min(e, starSi + charLen(d[starSi]));
                si = starSi;
            } else {
                return false;
            }
        }
        while (pi < m && kind[pi] == MANY) pi++;
        return pi == m;
    }
}
