package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

/** The string-function rewriter and byte kernels, over real Arrow vectors. */
class StringFnsTest {

    private static final TableSchema S = TableSchema.ofKinds(Map.of(
            "name", ColumnKind.STRING, "other", ColumnKind.STRING, "id", ColumnKind.INT32));

    private static final List<String> NAMES = java.util.Arrays.asList(
            "Alice", "  bob  ", null, "ÉCOLE", "", "Straße", "x", "列 ab", null, "MiXeD 123");
    private static final List<String> OTHERS = java.util.Arrays.asList(
            "-z", null, "q", "  ", "w", "ß", "", "é", null, "!");

    private static StringFns.Result rw(String sql) {
        return StringFns.rewrite(SqlParser.parse(sql), S, 0);
    }

    private static VarCharVector vec(RootAllocator a, String name, List<String> v) {
        VarCharVector x = new VarCharVector(name, a);
        x.allocateNew(Math.max(v.size(), 1));
        for (int i = 0; i < v.size(); i++) {
            if (v.get(i) == null) x.setNull(i); else x.setSafe(i, v.get(i).getBytes(StandardCharsets.UTF_8));
        }
        x.setValueCount(v.size());
        return x;
    }

    /** Evaluates the first derived string column of {@code sql} over NAMES/OTHERS. */
    private static List<String> eval(String sql) {
        StringFns.Result r = rw(sql);
        try (RootAllocator a = new RootAllocator();
             VarCharVector n = vec(a, "name", NAMES);
             VarCharVector o = vec(a, "other", OTHERS);
             DerivedColumns.Holder h = new DerivedColumns.Holder(r.columns())) {
            VectorSchemaRoot src = new VectorSchemaRoot(List.of(n, o));
            VectorSchemaRoot ext = h.extend(src, a);
            VarCharVector out = (VarCharVector) ext.getVector(r.columns().get(0).name);
            List<String> res = new ArrayList<>();
            for (int i = 0; i < NAMES.size(); i++) res.add(out.isNull(i) ? null : new String(out.get(i), StandardCharsets.UTF_8));
            return res;
        }
    }

    private static List<Double> lengths(String sql, int mask) {
        StringFns.Result r = rw(sql);
        try (RootAllocator a = new RootAllocator();
             VarCharVector n = vec(a, "name", NAMES);
             Float8Vector out = new Float8Vector("out", a)) {
            out.allocateNew(NAMES.size());
            r.masks().get(mask).eval(new VectorSchemaRoot(List.of(n)), NAMES.size(), out);
            List<Double> res = new ArrayList<>();
            for (int i = 0; i < NAMES.size(); i++) res.add(out.isNull(i) ? null : out.get(i));
            return res;
        }
    }

    private static String trimSpaces(String s) {
        int a = 0, b = s.length();
        while (a < b && s.charAt(a) == ' ') a++;
        while (b > a && s.charAt(b - 1) == ' ') b--;
        return s.substring(a, b);
    }

    private static String substr(String s, long start, long count, boolean has) {
        int[] cps = s.codePoints().toArray();
        long from = Math.max(start, 1), toEx = has ? start + count : Long.MAX_VALUE;
        StringBuilder sb = new StringBuilder();
        for (long p = from; p < toEx && p <= cps.length; p++) sb.appendCodePoint(cps[(int) p - 1]);
        return sb.toString();
    }

    @Test
    void lowerAndUpperFoldAsciiByByteAndUnicodeWhenNeeded() {
        List<String> lo = eval("SELECT LOWER(name) FROM t"), up = eval("SELECT UPPER(name) FROM t");
        for (int i = 0; i < NAMES.size(); i++) {
            String n = NAMES.get(i);
            assertEquals(n == null ? null : n.toLowerCase(Locale.ROOT), lo.get(i), "lower " + n);
            assertEquals(n == null ? null : n.toUpperCase(Locale.ROOT), up.get(i), "upper " + n);
        }
        assertEquals("STRASSE", up.get(5), "full Unicode mapping, not a per-byte guess");
    }

    @Test
    void trimRemovesSpacesOnlyAtBothEnds() {
        List<String> t = eval("SELECT TRIM(name) FROM t");
        for (int i = 0; i < NAMES.size(); i++) assertEquals(NAMES.get(i) == null ? null : trimSpaces(NAMES.get(i)), t.get(i));
        assertEquals("bob", t.get(1));
    }

    @Test
    void ltrimAndRtrimRemoveSpacesFromOneEndOnly() {
        List<String> l = eval("SELECT LTRIM(name) FROM t");
        List<String> r = eval("SELECT RTRIM(name) FROM t");
        for (int i = 0; i < NAMES.size(); i++) {
            String s = NAMES.get(i);
            assertEquals(s == null ? null : s.replaceFirst("^ +", ""), l.get(i));
            assertEquals(s == null ? null : s.replaceFirst(" +$", ""), r.get(i));
        }
        assertEquals("bob  ", l.get(1));
        assertEquals("  bob", r.get(1));
    }

    @Test
    void substringIsOneBasedByCharacterWithPostgresEdgeRules() {
        long[][] cases = {{1, 2}, {2, 3}, {0, 2}, {-1, 3}, {3, 0}, {5, 9}, {20, 2}, {2, 100}};
        for (long[] c : cases) {
            List<String> got = eval("SELECT SUBSTRING(name, " + (c[0] < 0 ? "- " + (-c[0]) : c[0]) + ", " + c[1] + ") FROM t");
            for (int i = 0; i < NAMES.size(); i++) {
                assertEquals(NAMES.get(i) == null ? null : substr(NAMES.get(i), c[0], c[1], true), got.get(i),
                        "SUBSTRING(" + NAMES.get(i) + ", " + c[0] + ", " + c[1] + ")");
            }
        }
        List<String> rest = eval("SELECT SUBSTR(name, 3) FROM t");
        for (int i = 0; i < NAMES.size(); i++) assertEquals(NAMES.get(i) == null ? null : substr(NAMES.get(i), 3, 0, false), rest.get(i));
        assertEquals("列 ab".substring(0, 1), eval("SELECT SUBSTRING(name, 1, 1) FROM t").get(7), "a multi-byte character is one character");
    }

    @Test
    void concatSkipsNullsButThePipeOperatorDoesNot() {
        List<String> c = eval("SELECT CONCAT(name, '-', other, 7) FROM t");
        List<String> p = eval("SELECT name || '-' || other FROM t");
        for (int i = 0; i < NAMES.size(); i++) {
            String n = NAMES.get(i), o = OTHERS.get(i);
            assertEquals((n == null ? "" : n) + "-" + (o == null ? "" : o) + "7", c.get(i));
            assertEquals(n == null || o == null ? null : n + "-" + o, p.get(i));
        }
    }

    @Test
    void expressionsNest() {
        List<String> got = eval("SELECT UPPER(TRIM(SUBSTRING(name || other, 2, 5))) FROM t");
        for (int i = 0; i < NAMES.size(); i++) {
            String n = NAMES.get(i), o = OTHERS.get(i);
            String e = n == null || o == null ? null : trimSpaces(substr(n + o, 2, 5, true)).toUpperCase(Locale.ROOT);
            assertEquals(e, got.get(i));
        }
    }

    @Test
    void lengthCountsCharactersAndOctetLengthCountsBytes() {
        List<Double> chars = lengths("SELECT LENGTH(name) FROM t", 0), bytes = lengths("SELECT OCTET_LENGTH(name) FROM t", 0);
        for (int i = 0; i < NAMES.size(); i++) {
            String n = NAMES.get(i);
            assertEquals(n == null ? null : (Double) (double) n.codePointCount(0, n.length()), chars.get(i), "length " + n);
            assertEquals(n == null ? null : (Double) (double) n.getBytes(StandardCharsets.UTF_8).length, bytes.get(i), "octets " + n);
        }
    }

    @Test
    void callsAreReplacedByDerivedColumnsAndOutputNamesAreKept() {
        var r = rw("SELECT LOWER(name) AS l, id FROM t WHERE LOWER(name) = 'abc' AND LENGTH(name) > 3 AND UPPER(name) LIKE 'A%'");
        assertEquals(2, r.columns().size(), "LOWER is shared, UPPER is the second");
        assertEquals(1, r.masks().size());
        assertEquals("l", r.stmt().items().get(0).alias());
        assertEquals(r.columns().get(0).name, r.stmt().items().get(0).exprText());
        assertEquals(ColumnKind.STRING, r.columns().get(0).kind);
        assertEquals(3, r.used());
        assertEquals("LOWER(name)", rw("SELECT LOWER(name) FROM t").stmt().items().get(0).alias(), "the original text names the column");
    }

    private static final TableSchema WITH_INT = TableSchema.ofKinds(Map.of("name", ColumnKind.STRING, "n", ColumnKind.INT32, "x", ColumnKind.FLOAT64));

    @Test
    void castToVarcharAcceptsStringExpressionsAndIntegerColumns() {
        var r = StringFns.rewrite(SqlParser.parse("SELECT CAST(n AS VARCHAR), CAST(UPPER(name) AS VARCHAR) FROM t"), WITH_INT, 0);
        assertEquals(2, r.columns().size());
        assertEquals("CAST(n AS VARCHAR)", r.stmt().items().get(0).alias());
        assertEquals(ColumnKind.STRING, r.columns().get(0).kind);
        ParquetSqlException e = assertThrows(ParquetSqlException.class,
                () -> StringFns.rewrite(SqlParser.parse("SELECT CAST(x AS VARCHAR) FROM t"), WITH_INT, 0));
        assertTrue(e.getMessage().contains("FLOAT64"), e.getMessage());
    }

    @Test
    void lengthWorksWhereANumberDoes() {
        var r = rw("SELECT LENGTH(name) + id AS z FROM t ORDER BY LENGTH(name) DESC");
        assertTrue(r.columns().isEmpty());
        assertEquals(1, r.masks().size(), "one mask for both uses");
        assertEquals(r.masks().get(0).name, r.stmt().orderBy().get(0).exprText());
    }

    @Test
    void statementsWithoutACallAreUntouchedAndLiteralsAreNotScanned() {
        var st = SqlParser.parse("SELECT name FROM t WHERE id > 1");
        assertSame(st, StringFns.rewrite(st, S, 0).stmt());
        assertTrue(rw("SELECT name FROM t WHERE name = 'LOWER(x) || y'").columns().isEmpty());
    }

    private static void assertNamed(String sql, String fragment) {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> rw(sql), sql);
        assertTrue(e.getMessage().contains(fragment), sql + " -> " + e.getMessage());
    }

    @Test
    void unsupportedUsesFailWithNamedErrors() {
        assertNamed("SELECT LOWER(id) FROM t", "INT32");
        assertNamed("SELECT LOWER(zzz) FROM t", "unknown column");
        assertNamed("SELECT id FROM t ORDER BY LOWER(name)", "ORDER BY");
        assertNamed("SELECT id FROM t GROUP BY UPPER(name)", "GROUP BY");
        assertNamed("SELECT COUNT(DISTINCT LOWER(name)) FROM t", "COUNT argument");
        assertNamed("SELECT LOWER(name) + 1 FROM t", "arithmetic");
        assertNamed("SELECT SUBSTRING(name, id, 2) FROM t", "integer literal");
        assertNamed("SELECT SUBSTRING(name, 1, - 2) FROM t", "negative");
        assertNamed("SELECT CONCAT(LENGTH(name), 'x') FROM t", "returns a number");
        assertNamed("SELECT NOSUCH(name) || 'x' FROM t", "not a string function");
    }

    @Test
    void aQuotedColumnIsEvaluatedByItsRealName() {
        TableSchema q = TableSchema.ofKinds(Map.of("my col", ColumnKind.STRING));
        var r = StringFns.rewrite(SqlParser.parse("SELECT UPPER(\"my col\") FROM t"), q, 0);
        assertEquals(1, r.columns().size());
        try (RootAllocator a = new RootAllocator();
             VarCharVector v = vec(a, "my col", List.of("ab", "c"));
             DerivedColumns.Holder h = new DerivedColumns.Holder(r.columns())) {
            VectorSchemaRoot ext = h.extend(new VectorSchemaRoot(List.of(v)), a);
            VarCharVector out = (VarCharVector) ext.getVector(r.columns().get(0).name);
            assertEquals("AB", new String(out.get(0), StandardCharsets.UTF_8));
            assertEquals("C", new String(out.get(1), StandardCharsets.UTF_8));
        }
    }
}
