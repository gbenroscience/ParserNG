package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ArrowSql;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Differential conformance test: the same SQL over the same data, run once through the in-memory
 * {@code ArrowQuery} (via {@link ArrowSql}) and once through the Parquet bridge ({@link ParquetSql}).
 *
 * <p><b>DRAFT. Not compiled or run when written.</b> It reuses the idioms of {@code ParquetFixtures} and
 * {@code ParquetSqlTest}; fix any API drift on first compile.
 *
 * <p>Two groups:
 * <ul>
 * <li>{@link #agree}: queries both engines must answer identically. A failure is a conformance bug.</li>
 * <li>{@code probe*}: places where the two are suspected to differ. The Parquet side is asserted against
 * the SQL-standard answer (computed by a small Java oracle or stated in the README). The ArrowQuery side is
 * only <em>reported</em> (printed as AGREE / DIVERGE), because its behaviour is currently undocumented.
 * Promote a probe to an assertion once you decide what ArrowQuery should do.</li>
 * </ul>
 * Every test closes its allocator, so an Arrow buffer leak in either engine fails the test.
 */
class ArrowQueryParquetSqlConformanceTest {

    private static final int N = 40;

    // ------------------------------------------------------------------ the data (one source of truth)

    /** id = i. */
    private static double id(int i) { return i; }

    /** a: null when i%5==0, else i%17 (1..16). */
    private static Double a(int i) { return i % 5 == 0 ? null : (double) (i % 17); }

    /** b: null when i%4==0, else (i%3)*10 (0, 10, 20). */
    private static Double b(int i) { return i % 4 == 0 ? null : (i % 3) * 10.0; }

    /** k: a clean group key, i%4 (no negative zero, no null). */
    private static double k(int i) { return i % 4; }

    /** g: group key containing both 0.0 and -0.0, and 1..5. */
    private static double g(int i) { return i % 6 == 0 ? (i % 12 == 0 ? -0.0 : 0.0) : i % 6; }

    /** fl: float32, null when i%9==0, else (i%10)*0.1f. Includes 0.1f exactly. */
    private static Float fl(int i) { return i % 9 == 0 ? null : (i % 10) * 0.1f; }

    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message d {\n"
            + "  required double id;\n"
            + "  optional double a;\n"
            + "  optional double b;\n"
            + "  required double k;\n"
            + "  required double g;\n"
            + "  optional float fl;\n"
            + "}");

    private static Path dir;
    private static Path file;

    @BeforeAll
    static void writeFile() throws IOException {
        dir = Files.createTempDirectory("conformance");
        file = dir.resolve("d.parquet");
        SimpleGroupFactory f = new SimpleGroupFactory(SCHEMA);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(SCHEMA)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
                .withDictionaryEncoding(false)
                .build()) {
            for (int i = 0; i < N; i++) {
                Group row = f.newGroup();
                row.append("id", id(i));
                if (a(i) != null) row.append("a", a(i));
                if (b(i) != null) row.append("b", b(i));
                row.append("k", k(i));
                row.append("g", g(i));
                if (fl(i) != null) row.append("fl", fl(i));
                w.write(row);
            }
        }
    }

    @AfterAll
    static void cleanUp() throws IOException {
        if (file != null) Files.deleteIfExists(file);
        if (dir != null) Files.deleteIfExists(dir);
    }

    /** The same data as an in-memory all-Float8 root, for ArrowQuery. */
    private static VectorSchemaRoot doubleRoot(RootAllocator alloc) {
        Float8Vector id = new Float8Vector("id", alloc), a = new Float8Vector("a", alloc), b = new Float8Vector("b", alloc),
                k = new Float8Vector("k", alloc), g = new Float8Vector("g", alloc);
        for (Float8Vector v : List.of(id, a, b, k, g)) v.allocateNew(N);
        for (int i = 0; i < N; i++) {
            id.set(i, id(i));
            if (a(i) == null) a.setNull(i); else a.set(i, a(i));
            if (b(i) == null) b.setNull(i); else b.set(i, b(i));
            k.set(i, k(i));
            g.set(i, g(i));
        }
        List<FieldVector> vecs = List.of(id, a, b, k, g);
        for (FieldVector v : vecs) v.setValueCount(N);
        VectorSchemaRoot r = new VectorSchemaRoot(vecs);
        r.setRowCount(N);
        return r;
    }

    /** id and fl as float32, for the float32 probe. */
    private static VectorSchemaRoot floatRoot(RootAllocator alloc) {
        Float4Vector id = new Float4Vector("id", alloc), fl = new Float4Vector("fl", alloc);
        id.allocateNew(N);
        fl.allocateNew(N);
        for (int i = 0; i < N; i++) {
            id.set(i, (float) id(i));
            if (fl(i) == null) fl.setNull(i); else fl.set(i, fl(i));
        }
        id.setValueCount(N);
        fl.setValueCount(N);
        VectorSchemaRoot r = new VectorSchemaRoot(List.of(id, fl));
        r.setRowCount(N);
        return r;
    }

    // ------------------------------------------------------------------ running and normalising

    /** Column names plus rows; numbers become doubles and -0.0 becomes 0.0 so values compare numerically. */
    private record Rows(List<String> names, List<List<Object>> rows) {
        @Override public String toString() { return names + " " + rows; }
    }

    private static Rows rowsOf(VectorSchemaRoot r) {
        List<String> names = new ArrayList<>();
        for (FieldVector v : r.getFieldVectors()) names.add(v.getName());
        List<List<Object>> rows = new ArrayList<>();
        for (int i = 0; i < r.getRowCount(); i++) {
            List<Object> row = new ArrayList<>();
            for (FieldVector v : r.getFieldVectors()) {
                Object o = v.isNull(i) ? null : v.getObject(i);
                if (o instanceof Number n) {
                    double d = n.doubleValue();
                    o = d == 0.0 ? 0.0 : d; // fold -0.0
                }
                row.add(o);
            }
            rows.add(row);
        }
        return new Rows(names, rows);
    }

    /** Without ORDER BY the group/row order is not defined, so both sides are sorted into a canonical order. */
    private static Rows canonical(Rows r, String sql) {
        if (sql.toUpperCase().contains("ORDER BY")) return r;
        List<List<Object>> sorted = new ArrayList<>(r.rows());
        sorted.sort(Comparator.comparing(Object::toString));
        return new Rows(r.names(), sorted);
    }

    private static Rows viaArrowQuery(RootAllocator alloc, VectorSchemaRoot root, String sql) {
        try (VectorSchemaRoot out = ArrowSql.execute(root, sql)) {
            return rowsOf(out);
        }
    }

    /** batchSize 7 makes the 40 rows span several batches, so streaming merge and top-N cross batch boundaries. */
    private static Rows viaParquetSql(RootAllocator alloc, String sql) {
        try (ParquetSqlPlan p = ParquetSql.plan(file, sql, ParquetSqlOptions.defaults().withBatchSize(7));
             VectorSchemaRoot out = p.execute(alloc)) {
            return rowsOf(out);
        }
    }

    // ------------------------------------------------------------------ group 1: must agree

    static Stream<Arguments> agree() {
        return Stream.of(
                Arguments.of("filter and", "SELECT id, a FROM t WHERE a > 10 AND b < 20"),
                Arguments.of("filter not", "SELECT id FROM t WHERE NOT (a > 10)"),
                Arguments.of("between", "SELECT id FROM t WHERE a BETWEEN 3 AND 9"),
                Arguments.of("not between", "SELECT id FROM t WHERE a NOT BETWEEN 3 AND 9"),
                Arguments.of("in", "SELECT id FROM t WHERE a IN (1, 5, 9, 13)"),
                Arguments.of("not in", "SELECT id FROM t WHERE a NOT IN (1, 5, 9, 13)"),
                Arguments.of("neq excludes null", "SELECT id FROM t WHERE a <> 3"),
                Arguments.of("is null", "SELECT id FROM t WHERE a IS NULL"),
                Arguments.of("is not null and", "SELECT id FROM t WHERE a IS NOT NULL AND b IS NOT NULL"),
                Arguments.of("computed null propagates", "SELECT id, a * 2 + b AS v FROM t"),
                Arguments.of("computed filter", "SELECT id FROM t WHERE a + b > 12"),
                Arguments.of("order nulls last desc", "SELECT id, a FROM t ORDER BY a DESC"),
                Arguments.of("order nulls last asc", "SELECT id, a FROM t ORDER BY a"),
                Arguments.of("order two keys", "SELECT id, a, b FROM t ORDER BY b, a DESC"),
                Arguments.of("top n", "SELECT id, a FROM t ORDER BY a DESC LIMIT 6"),
                Arguments.of("top n two keys", "SELECT id FROM t ORDER BY b DESC, id DESC LIMIT 9"),
                Arguments.of("top n larger than matches", "SELECT id FROM t WHERE a > 14 ORDER BY id DESC LIMIT 50"),
                Arguments.of("limit", "SELECT id FROM t LIMIT 5"),
                Arguments.of("global aggregates", "SELECT COUNT(*) AS c, COUNT(a) AS ca, SUM(a) AS s, AVG(a) AS av, MIN(a) AS mn, MAX(a) AS mx FROM t"),
                Arguments.of("global aggregates filtered", "SELECT COUNT(*) AS c, SUM(b) AS s, AVG(b) AS av FROM t WHERE a > 4"),
                Arguments.of("group by", "SELECT k, COUNT(*) AS c, SUM(a) AS s, AVG(b) AS av, MIN(a) AS mn, MAX(b) AS mx FROM t GROUP BY k"),
                Arguments.of("group by having", "SELECT k, SUM(a) AS s FROM t GROUP BY k HAVING s > 60"),
                Arguments.of("group by order limit", "SELECT k, COUNT(a) AS ca FROM t GROUP BY k ORDER BY ca DESC, k LIMIT 3"),
                Arguments.of("group by where", "SELECT k, COUNT(*) AS c FROM t WHERE b IS NOT NULL GROUP BY k")
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("agree")
    void agree(String name, String sqlOnT) {
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            Rows arrow = canonical(viaArrowQuery(alloc, root, sqlOnT), sqlOnT);
            Rows parquet = canonical(viaParquetSql(alloc, sqlOnT), sqlOnT);
            assertEquals(arrow, parquet, "ArrowQuery and ParquetSql disagree on: " + sqlOnT);
        }
    }

    // ------------------------------------------------------------------ group 2: suspected divergences

    /**
     * SQL three-valued logic: {@code a > 5 OR b > 5} keeps a row where a is null and b > 5. ParquetSql rewrites simple
     * numeric leaves under an OR into two-valued masks (README, "OR and nulls"); ArrowQuery propagates null through
     * the leaf, which would drop the row. The Parquet side is asserted; the ArrowQuery side is reported.
     */
    @Test
    void probeOrWithNulls() {
        String sql = "SELECT id FROM t WHERE a > 5 OR b > 5";
        List<List<Object>> expected = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            Boolean x = a(i) == null ? null : a(i) > 5;
            Boolean y = b(i) == null ? null : b(i) > 5;
            if (Boolean.TRUE.equals(x) || Boolean.TRUE.equals(y)) expected.add(List.of((double) i));
        }
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            Rows parquet = viaParquetSql(alloc, sql);
            assertEquals(expected, parquet.rows(), "ParquetSql must follow SQL 3VL for OR");
            report("OR with nulls", sql, () -> viaArrowQuery(alloc, root, sql), parquet);
        }
    }

    /**
     * Characterises the README caveat ("OR and nulls"): a numeric leaf that is an expression ({@code a + 1 > 6}) is
     * evaluated with null propagation, so when {@code a} is null the OR is dropped even if {@code b > 5} is true
     * (ids 5, 10, 25, 35). A row where {@code a} is present is correct. This pins the documented behaviour; if the
     * limitation is ever lifted, change the expectation to full three-valued logic.
     */
    @Test
    void probeOrWithExpressionLeafDocumentedLimitation() {
        String sql = "SELECT id FROM t WHERE a + 1 > 6 OR b > 5";
        List<List<Object>> expected = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            if (a(i) != null && (a(i) + 1 > 6 || (b(i) != null && b(i) > 5))) expected.add(List.of((double) i));
        }
        expected.sort(Comparator.comparing(Object::toString));
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            Rows parquet = canonical(viaParquetSql(alloc, sql), sql);
            assertEquals(expected, parquet.rows(), "ParquetSql should match its documented expression-leaf behaviour");
            report("OR with expression leaf (documented limitation)", sql, () -> viaArrowQuery(alloc, root, sql), parquet);
        }
    }

    private static Boolean and3(Boolean x, Boolean y) {
        if (Boolean.FALSE.equals(x) || Boolean.FALSE.equals(y)) return false;
        if (x == null || y == null) return null;
        return true;
    }

    /**
     * {@code NOT (x AND y)} with x false and y null is TRUE in SQL (false AND null = false). The engine rewrites it
     * to {@code a <= 10 OR b <= 5}, so this is the same OR/null difference reached through NOT.
     */
    @Test
    void probeNotOverAndWithNulls() {
        String sql = "SELECT id FROM t WHERE NOT (a > 10 AND b > 5)";
        List<List<Object>> expected = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            Boolean x = a(i) == null ? null : a(i) > 10;
            Boolean y = b(i) == null ? null : b(i) > 5;
            Boolean and = and3(x, y);
            if (and != null && !and) expected.add(List.of((double) i));
        }
        expected.sort(Comparator.comparing(Object::toString));
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            Rows parquet = canonical(viaParquetSql(alloc, sql), sql);
            assertEquals(expected, parquet.rows(), "ParquetSql must follow SQL 3VL for NOT over AND");
            report("NOT over AND with nulls", sql, () -> viaArrowQuery(alloc, root, sql), parquet);
        }
    }

    /**
     * HAVING with an aggregate call instead of the alias. README documents the alias form for ArrowQuery and the
     * aggregate form for ParquetSql. ArrowQuery currently fails at execution with a binding error.
     */
    @Test
    void probeHavingAggregateCall() {
        String sql = "SELECT k, SUM(a) AS s FROM t GROUP BY k HAVING SUM(a) > 60";
        java.util.Map<Double, Double> sums = new java.util.TreeMap<>();
        for (int i = 0; i < N; i++) if (a(i) != null) sums.merge(k(i), a(i), Double::sum);
        List<List<Object>> expected = new ArrayList<>();
        sums.forEach((key, sum) -> { if (sum > 60) expected.add(List.of(key, sum)); });
        expected.sort(Comparator.comparing(Object::toString));
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            Rows parquet = canonical(viaParquetSql(alloc, sql), sql);
            assertEquals(expected, parquet.rows(), "ParquetSql: HAVING with an aggregate call");
            report("HAVING SUM(a) > 60", sql, () -> viaArrowQuery(alloc, root, sql), parquet);
        }
    }

    /** An empty global aggregate: SQL (and the README) say COUNT is 0 and SUM/MIN are null, in one row. */
    @Test
    void probeEmptyGlobalAggregate() {
        String sql = "SELECT COUNT(*) AS c, SUM(a) AS s, MIN(a) AS mn FROM t WHERE a > 1000";
        List<Object> row = new ArrayList<>();
        row.add(0.0);
        row.add(null);
        row.add(null);
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            Rows parquet = viaParquetSql(alloc, sql);
            assertEquals(List.of(row), parquet.rows(), "ParquetSql: empty global aggregate");
            report("empty global aggregate", sql, () -> viaArrowQuery(alloc, root, sql), parquet);
        }
    }

    /** g holds both 0.0 and -0.0 (i%12==0 gives -0.0). SQL has one group for them; ParquetSql merges them via norm(). */
    @Test
    void probeNegativeZeroGroupKey() {
        String sql = "SELECT g, COUNT(*) AS c FROM t GROUP BY g";
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            Rows parquet = viaParquetSql(alloc, sql);
            assertEquals(6, parquet.rows().size(), "keys 0 (merged with -0), 1, 2, 3, 4, 5");
            report("-0.0 group key", sql, () -> viaArrowQuery(alloc, root, sql), parquet);
        }
    }

    /**
     * float32 column against a literal that is not exactly representable. ParquetSql widens to double, so 0.1f
     * (0.10000000149...) is greater than 0.1; a float32 kernel would compare 0.1f > 0.1f and say false.
     * The Parquet side is asserted against the widened-double answer; the float32 ArrowQuery side is reported.
     */
    @Test
    void probeFloat32Literal() {
        String sql = "SELECT id FROM t WHERE fl > 0.1";
        List<List<Object>> expected = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            if (fl(i) != null && (double) fl(i) > 0.1) expected.add(List.of((double) i));
        }
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = floatRoot(alloc)) {
            Rows parquet = viaParquetSql(alloc, sql);
            assertEquals(expected, parquet.rows(), "ParquetSql compares float32 columns as doubles");
            report("float32 vs 0.1", sql, () -> viaArrowQuery(alloc, root, sql), parquet);
        }
    }

    // ------------------------------------------------------------------ regressions (ArrowQuery)

    /** FAILS on the current code: a global aggregate over zero rows must still return one row (COUNT 0, others null). */
    @Test
    void regressionArrowQueryEmptyGlobalAggregateReturnsOneRow() {
        String sql = "SELECT COUNT(*) AS c, SUM(a) AS s FROM t WHERE a > 1000";
        List<Object> row = new ArrayList<>();
        row.add(0.0);
        row.add(null);
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            assertEquals(List.of(row), viaArrowQuery(alloc, root, sql).rows());
        }
    }

    /** FAILS on the current code: 0.0 and -0.0 are the same SQL value and must fall into one group. */
    @Test
    void regressionArrowQueryNegativeZeroIsOneGroup() {
        String sql = "SELECT g, COUNT(*) AS c FROM t GROUP BY g";
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            assertEquals(6, viaArrowQuery(alloc, root, sql).rows().size());
        }
    }

    /** FAILS before the fix: SELECT DISTINCT over an empty global aggregate must still see its one row. */
    @Test
    void regressionArrowQueryDistinctOverEmptyGlobalAggregate() {
        String sql = "SELECT DISTINCT COUNT(*) AS c FROM t WHERE a > 1000";
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            assertEquals(List.of(List.of(0.0)), viaArrowQuery(alloc, root, sql).rows());
        }
    }

    /** Global aggregate over a NON-empty input must be unchanged by the empty-input seeding (guards the hot path). */
    @Test
    void regressionArrowQueryNonEmptyGlobalAggregateUnchanged() {
        String sql = "SELECT COUNT(*) AS c, COUNT(a) AS ca, SUM(a) AS s FROM t";
        double sum = 0;
        int ca = 0;
        for (int i = 0; i < N; i++) if (a(i) != null) { sum += a(i); ca++; }
        try (RootAllocator alloc = new RootAllocator(); VectorSchemaRoot root = doubleRoot(alloc)) {
            assertEquals(List.of(List.of((double) N, (double) ca, sum)), viaArrowQuery(alloc, root, sql).rows());
        }
    }

    // ------------------------------------------------------------------ reporting

    private static void report(String label, String sql, java.util.function.Supplier<Rows> arrowSide, Rows parquet) {
        try {
            Rows arrow = canonical(arrowSide.get(), sql);
            Rows p = canonical(parquet, sql);
            System.out.println("[conformance] " + (arrow.equals(p) ? "AGREE   " : "DIVERGE ") + label + "\n  " + sql
                    + "\n  ArrowQuery: " + arrow + "\n  ParquetSql: " + p);
        } catch (RuntimeException e) {
            System.out.println("[conformance] ARROWQUERY THREW " + label + "\n  " + sql + "\n  " + e);
        }
    }
}