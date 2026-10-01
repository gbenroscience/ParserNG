package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.ast.*;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Executes the native predicate kernels on real Arrow vectors, with no Parquet file involved, and
 * checks each result against a plain-Java reference model of the same rows. Covers nulls, a string
 * longer than the kernels' scratch buffer, a multibyte character for {@code _}, all boolean forms,
 * dates, and timestamps across zones and sub-unit precision.
 */
class NativeLeavesKernelTest {

    private static final String LONG = "a".repeat(100) + "xyz";
    private static final String[] NAME = {"apple", "banana", null, "cherry", "apple pie", "", "\u00c9clair", "banana split", LONG};
    private static final Boolean[] FLAG = {true, false, null, true, false, null, true, false, true};
    private static final Integer[] DAY = {19000, 19001, null, 19003, 19004, 19005, 19006, 19007, 19008};
    private static final long BASE_SEC = 1_700_000_000L;
    private static final Long[] TS_SEC = {BASE_SEC, BASE_SEC + 1, null, BASE_SEC + 3, BASE_SEC + 4, BASE_SEC + 5, BASE_SEC + 6, BASE_SEC + 7, BASE_SEC + 8};
    private static final int N = NAME.length;

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static String ts(long epochSecond) {
        return LocalDateTime.ofEpochSecond(epochSecond, 0, ZoneOffset.UTC).format(FMT);
    }

    private static String date(int epochDay) {
        return java.time.LocalDate.ofEpochDay(epochDay).toString();
    }

    private final BufferAllocator alloc = new RootAllocator();
    private final VectorSchemaRoot root = build();
    private final TableSchema schema = TableSchema.of(new Schema(List.of(
            new Field("name", FieldType.nullable(ArrowType.Utf8.INSTANCE), List.of()),
            new Field("flag", FieldType.nullable(ArrowType.Bool.INSTANCE), List.of()),
            new Field("day", FieldType.nullable(new ArrowType.Date(DateUnit.DAY)), List.of()),
            new Field("ts", FieldType.nullable(new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC")), List.of()),
            new Field("tsn", FieldType.nullable(new ArrowType.Timestamp(TimeUnit.MICROSECOND, null)), List.of()))));

    private VectorSchemaRoot build() {
        VarCharVector name = new VarCharVector("name", alloc);
        name.allocateNew();
        BitVector flag = new BitVector("flag", alloc);
        flag.allocateNew(N);
        DateDayVector day = new DateDayVector("day", alloc);
        day.allocateNew(N);
        TimeStampMicroTZVector ts = new TimeStampMicroTZVector("ts", alloc, "UTC");
        ts.allocateNew(N);
        TimeStampMicroVector tsn = new TimeStampMicroVector("tsn", alloc);
        tsn.allocateNew(N);
        for (int i = 0; i < N; i++) {
            if (NAME[i] == null) name.setNull(i); else name.setSafe(i, NAME[i].getBytes(StandardCharsets.UTF_8));
            if (FLAG[i] == null) flag.setNull(i); else flag.setSafe(i, FLAG[i] ? 1 : 0);
            if (DAY[i] == null) day.setNull(i); else day.setSafe(i, DAY[i]);
            if (TS_SEC[i] == null) { ts.setNull(i); tsn.setNull(i); }
            else { ts.setSafe(i, TS_SEC[i] * 1_000_000L); tsn.setSafe(i, TS_SEC[i] * 1_000_000L); }
        }
        name.setValueCount(N);
        flag.setValueCount(N);
        day.setValueCount(N);
        ts.setValueCount(N);
        tsn.setValueCount(N);
        List<FieldVector> vs = new ArrayList<>(List.of(name, flag, day, ts, tsn));
        return new VectorSchemaRoot(vs);
    }

    private interface Row { boolean test(int i); }

    /** Rewrites {@code cond}, runs every mask kernel, and combines the tree the way the engine would. */
    private boolean[] run(String cond) {
        BoolExpr w = SqlParser.parse("SELECT name FROM t WHERE " + cond).where();
        NativeLeaves.Result r = NativeLeaves.rewrite(w, schema);
        double[][] vals = new double[r.masks().size()][N];
        for (int m = 0; m < vals.length; m++) {
            Float8Vector out = new Float8Vector(r.masks().get(m).name, alloc);
            out.allocateNew(N);
            r.masks().get(m).eval(root, N, out);
            for (int i = 0; i < N; i++) {
                assertFalse(out.isNull(i), "masks are two-valued: never null");
                vals[m][i] = out.get(i);
            }
        }
        boolean[] res = new boolean[N];
        List<String> names = new ArrayList<>();
        for (NativeLeaves.Mask mk : r.masks()) names.add(mk.name);
        for (int i = 0; i < N; i++) res[i] = eval(r.where(), vals, names, i);
        return res;
    }

    private static boolean eval(BoolExpr e, double[][] vals, List<String> names, int row) {
        if (e instanceof AndExpr a) return eval(a.left(), vals, names, row) && eval(a.right(), vals, names, row);
        if (e instanceof OrExpr o) return eval(o.left(), vals, names, row) || eval(o.right(), vals, names, row);
        ComparisonExpr c = (ComparisonExpr) e;
        return vals[names.indexOf(c.left())][row] == 1.0;
    }

    private void check(String cond, Row expected) {
        boolean[] got = run(cond);
        for (int i = 0; i < N; i++) assertEquals(expected.test(i), got[i], cond + " at row " + i);
    }

    // ------------------------------------------------------------------ strings

    @Test
    void stringComparisons() {
        check("name = 'apple'", i -> "apple".equals(NAME[i]));
        check("name <> 'apple'", i -> NAME[i] != null && !NAME[i].equals("apple"));
        check("name = ''", i -> "".equals(NAME[i]));
        check("name > 'banana'", i -> NAME[i] != null && cmp(NAME[i], "banana") > 0);
        check("name >= 'banana'", i -> NAME[i] != null && cmp(NAME[i], "banana") >= 0);
        check("name < 'b'", i -> NAME[i] != null && cmp(NAME[i], "b") < 0);
        check("name <= 'apple'", i -> NAME[i] != null && cmp(NAME[i], "apple") <= 0);
        check("'banana' < name", i -> NAME[i] != null && cmp(NAME[i], "banana") > 0);
        check("name = '" + LONG + "'", i -> LONG.equals(NAME[i]));
    }

    /** Unsigned UTF-8 byte order, the documented ordering. */
    private static int cmp(String a, String b) {
        byte[] x = a.getBytes(StandardCharsets.UTF_8), y = b.getBytes(StandardCharsets.UTF_8);
        for (int k = 0; k < Math.min(x.length, y.length); k++) if ((x[k] & 0xFF) != (y[k] & 0xFF)) return (x[k] & 0xFF) - (y[k] & 0xFF);
        return x.length - y.length;
    }

    @Test
    void stringInBetweenAndNulls() {
        check("name IN ('apple', 'cherry', 'zzz')", i -> "apple".equals(NAME[i]) || "cherry".equals(NAME[i]));
        check("name NOT IN ('apple')", i -> NAME[i] != null && !NAME[i].equals("apple"));
        check("name IN ('apple', NULL)", i -> "apple".equals(NAME[i]));
        check("name NOT IN ('apple', NULL)", i -> false);
        check("name BETWEEN 'apple' AND 'banana'", i -> NAME[i] != null && cmp(NAME[i], "apple") >= 0 && cmp(NAME[i], "banana") <= 0);
        check("name NOT BETWEEN 'apple' AND 'banana'", i -> NAME[i] != null && (cmp(NAME[i], "apple") < 0 || cmp(NAME[i], "banana") > 0));
        check("name IS NULL", i -> NAME[i] == null);
        check("name IS NOT NULL", i -> NAME[i] != null);
        check("name = NULL", i -> false);
    }

    @Test
    void likeFamily() {
        check("name LIKE 'apple%'", i -> NAME[i] != null && NAME[i].startsWith("apple"));
        check("name NOT LIKE 'apple%'", i -> NAME[i] != null && !NAME[i].startsWith("apple"));
        check("name LIKE '%an%'", i -> NAME[i] != null && NAME[i].contains("an"));
        check("name LIKE '%xyz'", i -> NAME[i] != null && NAME[i].endsWith("xyz"));
        check("name LIKE ''", i -> "".equals(NAME[i]));
        check("name LIKE '_clair'", i -> "\u00c9clair".equals(NAME[i]));
        check("name LIKE 'b%a%a'", i -> "banana".equals(NAME[i]));
        check("name ILIKE 'APPLE%'", i -> NAME[i] != null && NAME[i].startsWith("apple"));
        check("name LIKE 'apple!%' ESCAPE '!'", i -> false);
    }

    @Test
    void regexp() {
        check("name REGEXP '^b'", i -> NAME[i] != null && NAME[i].startsWith("b"));
        check("name NOT REGEXP '^b'", i -> NAME[i] != null && !NAME[i].startsWith("b"));
        check("name RLIKE 'an{2}'", i -> false);
        check("name RLIKE 'a.a'", i -> NAME[i] != null && NAME[i].matches("(?s).*a.a.*"));
    }

    // ------------------------------------------------------------------ booleans

    @Test
    void booleanForms() {
        check("flag", i -> Boolean.TRUE.equals(FLAG[i]));
        check("flag = TRUE", i -> Boolean.TRUE.equals(FLAG[i]));
        check("NOT flag", i -> Boolean.FALSE.equals(FLAG[i]));
        check("flag = FALSE", i -> Boolean.FALSE.equals(FLAG[i]));
        check("flag <> TRUE", i -> Boolean.FALSE.equals(FLAG[i]));
        check("flag IS TRUE", i -> Boolean.TRUE.equals(FLAG[i]));
        check("flag IS FALSE", i -> Boolean.FALSE.equals(FLAG[i]));
        check("flag IS NOT TRUE", i -> !Boolean.TRUE.equals(FLAG[i]));
        check("flag IS NOT FALSE", i -> !Boolean.FALSE.equals(FLAG[i]));
        check("flag IS NULL", i -> FLAG[i] == null);
        check("flag IN (TRUE)", i -> Boolean.TRUE.equals(FLAG[i]));
        check("flag NOT IN (TRUE)", i -> Boolean.FALSE.equals(FLAG[i]));
    }

    // ------------------------------------------------------------------ dates

    @Test
    void dates() {
        check("day = DATE '" + date(19003) + "'", i -> DAY[i] != null && DAY[i] == 19003);
        check("day = '" + date(19003) + "'", i -> DAY[i] != null && DAY[i] == 19003);
        check("day <> DATE '" + date(19003) + "'", i -> DAY[i] != null && DAY[i] != 19003);
        check("day >= DATE '" + date(19004) + "'", i -> DAY[i] != null && DAY[i] >= 19004);
        check("day < DATE '" + date(19004) + "'", i -> DAY[i] != null && DAY[i] < 19004);
        check("day BETWEEN DATE '" + date(19001) + "' AND DATE '" + date(19004) + "'", i -> DAY[i] != null && DAY[i] >= 19001 && DAY[i] <= 19004);
        check("day IN (DATE '" + date(19000) + "', '" + date(19008) + "')", i -> DAY[i] != null && (DAY[i] == 19000 || DAY[i] == 19008));
        check("day NOT IN (DATE '" + date(19000) + "')", i -> DAY[i] != null && DAY[i] != 19000);
        check("day IS NULL", i -> DAY[i] == null);
    }

    // ------------------------------------------------------------------ timestamps

    @Test
    void timestampsAcrossZonesAndPrecision() {
        String t4 = ts(BASE_SEC + 4);
        check("ts >= TIMESTAMP '" + t4 + "'", i -> TS_SEC[i] != null && TS_SEC[i] >= BASE_SEC + 4);
        check("ts >= TIMESTAMP '" + t4 + "Z'", i -> TS_SEC[i] != null && TS_SEC[i] >= BASE_SEC + 4);
        check("ts = TIMESTAMP '" + ts(BASE_SEC + 4 + 7200) + "+02:00'", i -> TS_SEC[i] != null && TS_SEC[i] == BASE_SEC + 4);
        check("ts = TIMESTAMP '" + t4 + "'", i -> TS_SEC[i] != null && TS_SEC[i] == BASE_SEC + 4);
        check("ts <> TIMESTAMP '" + t4 + "'", i -> TS_SEC[i] != null && TS_SEC[i] != BASE_SEC + 4);
        check("ts < '" + t4 + "'", i -> TS_SEC[i] != null && TS_SEC[i] < BASE_SEC + 4);
        check("ts BETWEEN TIMESTAMP '" + ts(BASE_SEC + 1) + "' AND TIMESTAMP '" + ts(BASE_SEC + 5) + "'",
                i -> TS_SEC[i] != null && TS_SEC[i] >= BASE_SEC + 1 && TS_SEC[i] <= BASE_SEC + 5);
        check("ts IN (TIMESTAMP '" + ts(BASE_SEC) + "', TIMESTAMP '" + ts(BASE_SEC + 8) + "')",
                i -> TS_SEC[i] != null && (TS_SEC[i] == BASE_SEC || TS_SEC[i] == BASE_SEC + 8));
        // Half a second past a tick: no microsecond equals it, ordering stays exact.
        check("ts = TIMESTAMP '" + t4 + ".5'", i -> false);
        check("ts > TIMESTAMP '" + t4 + ".5'", i -> TS_SEC[i] != null && TS_SEC[i] >= BASE_SEC + 5);
        check("ts < TIMESTAMP '" + t4 + ".5'", i -> TS_SEC[i] != null && TS_SEC[i] <= BASE_SEC + 4);
        check("ts >= DATE '1970-01-01'", i -> TS_SEC[i] != null);
        check("ts IS NULL", i -> TS_SEC[i] == null);
    }

    @Test
    void zonedLiteralOnANaiveTimestampColumnIsRejected() {
        String t = ts(BASE_SEC + 4);
        check("tsn >= TIMESTAMP '" + t + "'", i -> TS_SEC[i] != null && TS_SEC[i] >= BASE_SEC + 4);
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> run("tsn >= TIMESTAMP '" + t + "Z'"));
        assertTrue(e.getMessage().contains("'tsn'") && e.getMessage().contains("ambiguous"), e.getMessage());
    }

    // ------------------------------------------------------------------ composition

    @Test
    void masksComposeUnderAndOr() {
        check("name = 'apple' OR flag IS NULL", i -> "apple".equals(NAME[i]) || FLAG[i] == null);
        check("name LIKE 'b%' AND flag IS FALSE", i -> NAME[i] != null && NAME[i].startsWith("b") && Boolean.FALSE.equals(FLAG[i]));
        check("(name IS NULL OR name = '') AND day IS NOT NULL", i -> (NAME[i] == null || NAME[i].isEmpty()) && DAY[i] != null);
    }

    @Test
    void emptyBatchIsFine() {
        BoolExpr w = SqlParser.parse("SELECT name FROM t WHERE name LIKE 'a%'").where();
        NativeLeaves.Result r = NativeLeaves.rewrite(w, schema);
        Float8Vector out = new Float8Vector("m", alloc);
        out.allocateNew(1);
        r.masks().get(0).eval(root, 0, out);
    }
}
