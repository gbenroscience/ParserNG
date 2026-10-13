package com.github.gbenroscience.sqlv1.demo;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import com.github.gbenroscience.parser.ng.parquet.v1.ScanMetrics;
import com.github.gbenroscience.sqlv1.parquet.ParquetSql;
import com.github.gbenroscience.sqlv1.parquet.ParquetSqlOptions;
import com.github.gbenroscience.sqlv1.parquet.ParquetSqlPlan;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.api.ReadSupport;
import org.apache.parquet.hadoop.example.GroupReadSupport;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalInputFile;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.function.ToDoubleFunction;

/**
 * Benchmarks {@code ParquetSql} against what it is built on, and says where the time goes.
 *
 * <h2>What is compared</h2>
 * Every workload is one SQL query and runs three ways over the <em>same file</em>:
 * <ol>
 * <li><b>sql</b>: {@code ParquetSql}, planned once and executed repeatedly (planning is timed separately).</li>
 * <li><b>direct</b>: the same answer computed by hand-written Java straight on parser-ng-parquet's Arrow batches
 * ({@link ParquetScan}, with the equivalent {@code select}/{@code pushdown}). This is the floor the SQL layer can
 * be measured against: {@code sql / direct} is the overhead of parsing, rewriting, widening, masks, the engine,
 * folding and gathering on top of the scan.</li>
 * <li><b>raw</b> (a few workloads): the parquet-java {@code Group} reader, the baseline of the "3x to 5x faster than
 * raw parquet-java" claim, so {@code raw / sql} shows what the claim is worth with SQL on top.</li>
 * </ol>
 * The three must agree on the answer (a checksum, relative tolerance 1e-9); a disagreement is reported and the run
 * exits non-zero, so the benchmark doubles as a correctness check at scale. Each workload also checks that the
 * allocator is empty afterwards (a leak is reported).
 *
 * <h2>Methodology</h2>
 * Borrowed from the other benchmarks of this project: fixed-seed data written before any timing and a warm page
 * cache (the parser-ng-parquet fair comparison); warm-up then repeated reps with the median, min and max, a
 * calibrated number of calls per rep, and every answer consumed so the JIT cannot drop the work ({@link Benchmark},
 * {@link ExecBench}). Added: bytes allocated per call by the calling thread (a parallel scan's workers are not
 * counted) and GC time per call, because the target is "as zero-alloc as possible".
 *
 * <h2>Profiling</h2>
 * {@code --profile <workload>} runs one workload for {@code --seconds} under an in-process sampler
 * ({@link StackSampler}) and prints self and inclusive time per layer (bridge, SQL engine, scan, expression engine,
 * Arrow, parquet-java, JDK) and the hottest methods. Use JFR or async-profiler for line-level detail.
 *
 * <h2>Usage</h2>
 * <pre>
 * java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED ... \
 *      com.github.gbenroscience.sqlv1.demo.ParquetSqlBenchmark [options]
 *   --rows N          rows in the file (default 3000000)
 *   --codec C         UNCOMPRESSED (default) or SNAPPY
 *   --rowgroup-mb N   target row group size (default 16)
 *   --batch N         batch size for both sql and direct (default: parser-ng-parquet's)
 *   --parallelism N   row groups decoded at once (default: parser-ng-parquet's)
 *   --only TEXT       only workloads whose name contains TEXT
 *   --no-raw          skip the parquet-java Group baselines (they are slow)
 *   --warmup N --reps N --min-rep-ms X   timing controls (defaults 3, 7, 300)
 *   --explain         print the plan of every workload
 *   --profile TEXT --seconds S   profile the first matching workload
 *   --csv FILE        also write the table as CSV
 *   --keep            keep the generated file (it is in the temp directory)
 * </pre>
 */
public final class ParquetSqlBenchmark {

    // ------------------------------------------------------------------ configuration

    private static final class Cfg {
        int rows = 3_000_000;
        CompressionCodecName codec = CompressionCodecName.UNCOMPRESSED;
        int rowGroupMb = 16;
        int batch = 0;
        int parallelism = 0;
        String only = null;
        boolean raw = true;
        int warmup = 3, reps = 7;
        double minRepMs = 300;
        boolean explain = false;
        String profile = null;
        int seconds = 10;
        String csv = null;
        boolean keep = false;
        Path file;
    }

    private static Cfg parse(String[] a) {
        Cfg c = new Cfg();
        for (int i = 0; i < a.length; i++) {
            switch (a[i]) {
                case "--rows" -> c.rows = Integer.parseInt(a[++i]);
                case "--codec" -> c.codec = CompressionCodecName.valueOf(a[++i].toUpperCase(Locale.ROOT));
                case "--rowgroup-mb" -> c.rowGroupMb = Integer.parseInt(a[++i]);
                case "--batch" -> c.batch = Integer.parseInt(a[++i]);
                case "--parallelism" -> c.parallelism = Integer.parseInt(a[++i]);
                case "--only" -> c.only = a[++i];
                case "--no-raw" -> c.raw = false;
                case "--warmup" -> c.warmup = Integer.parseInt(a[++i]);
                case "--reps" -> c.reps = Integer.parseInt(a[++i]);
                case "--min-rep-ms" -> c.minRepMs = Double.parseDouble(a[++i]);
                case "--explain" -> c.explain = true;
                case "--profile" -> c.profile = a[++i];
                case "--seconds" -> c.seconds = Integer.parseInt(a[++i]);
                case "--csv" -> c.csv = a[++i];
                case "--keep" -> c.keep = true;
                default -> throw new IllegalArgumentException("Unknown option " + a[i] + " (see the class comment)");
            }
        }
        return c;
    }

    private static ParquetSqlOptions options(Cfg c) {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withMetrics(true);
        if (c.batch > 0) o = o.withBatchSize(c.batch);
        if (c.parallelism > 0) o = o.withParallelism(c.parallelism);
        return o;
    }

    private static ParquetScan scan(Cfg c, String... cols) {
        ParquetScan s = ParquetScan.scan(c.file);
        if (cols.length > 0) s = s.select(cols);
        if (c.batch > 0) s = s.batchSize(c.batch);
        if (c.parallelism > 0) s = s.parallelism(c.parallelism);
        return s;
    }

    // ------------------------------------------------------------------ workloads

    /** A direct computation over parser-ng-parquet batches. */
    private interface Direct {
        double run(Cfg c, BufferAllocator a) throws Exception;
    }

    /** A computation with the parquet-java Group reader. */
    private interface Raw {
        double run(Cfg c) throws Exception;
    }

    private record Workload(String name, String note, String sql, ToDoubleFunction<VectorSchemaRoot> answer, Direct direct, Raw raw) { }

    private static final double SPLIT = 1e7; // separates a group count from a total in a checksum

    private static List<Workload> workloads(Cfg c) {
        long cutoff = (long) (c.rows * 0.9);
        List<Workload> w = new ArrayList<>();

        w.add(new Workload("count-star", "footer row count",
                "SELECT COUNT(*) AS c FROM t",
                r -> ((Float8Vector) r.getFieldVectors().get(0)).get(0),
                (cfg, a) -> ParquetFileInfo.read(cfg.file).rowCount(), null));

        w.add(new Workload("footer-stats", "COUNT/MIN/MAX from footer statistics",
                "SELECT MIN(id) AS lo, MAX(id) AS hi, COUNT(qty) AS cq FROM t",
                r -> col0(r, 0) + col0(r, 1) + col0(r, 2),
                (cfg, a) -> {
                    ParquetFileInfo info = ParquetFileInfo.read(cfg.file);
                    double lo = Double.MAX_VALUE, hi = -Double.MAX_VALUE, nn = 0;
                    for (var rg : info.rowGroups()) {
                        for (var cc : rg.columns()) {
                            if (cc.path().equals("id")) {
                                lo = Math.min(lo, Long.parseLong(cc.min()));
                                hi = Math.max(hi, Long.parseLong(cc.max()));
                            } else if (cc.path().equals("qty")) {
                                nn += rg.rowCount() - cc.nullCount();
                            }
                        }
                    }
                    return lo + hi + nn;
                }, null));

        w.add(new Workload("scan-project", "pure projection, zero-copy",
                "SELECT id, value FROM t",
                r -> sumLong(r, 0) + sumDouble(r, 1),
                (cfg, a) -> {
                    double s = 0;
                    try (ParquetBatchReader rd = scan(cfg, "id", "value").open(a)) {
                        while (rd.next()) s += sumLong(rd.root(), "id") + sumDouble(rd.root(), "value");
                    }
                    return s;
                },
                cfg -> {
                    double s = 0;
                    try (ParquetReader<Group> rd = groupReader(cfg.file)) {
                        Group g;
                        while ((g = rd.read()) != null) {
                            s += g.getLong("id", 0);
                            if (g.getFieldRepetitionCount("value") > 0) s += g.getDouble("value", 0);
                        }
                    }
                    return s;
                }));

        w.add(new Workload("filter-pruned", "WHERE id >= 90th percentile: row groups skipped by statistics",
                "SELECT id, value FROM t WHERE id >= " + cutoff,
                r -> r.getRowCount() * SPLIT + sumLong(r, 0),
                (cfg, a) -> {
                    long n = 0;
                    double s = 0;
                    try (ParquetBatchReader rd = scan(cfg, "id", "value").pushdown(Predicate.ge("id", cutoff)).open(a)) {
                        while (rd.next()) {
                            BigIntVector id = (BigIntVector) rd.root().getVector("id");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) {
                                if (id.get(i) >= cutoff) { n++; s += id.get(i); }
                            }
                        }
                    }
                    return n * SPLIT + s;
                }, null));

        w.add(new Workload("filter-numeric", "two numeric predicates, count; nulls drop the row",
                "SELECT COUNT(*) AS c FROM t WHERE value > 0.5 AND qty < 50",
                r -> col0(r, 0),
                (cfg, a) -> {
                    long n = 0;
                    try (ParquetBatchReader rd = scan(cfg, "value", "qty")
                            .pushdown(Predicate.and(Predicate.gt("value", 0.5), Predicate.lt("qty", 50))).open(a)) {
                        while (rd.next()) {
                            Float8Vector v = (Float8Vector) rd.root().getVector("value");
                            IntVector q = (IntVector) rd.root().getVector("qty");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) {
                                if (!v.isNull(i) && !q.isNull(i) && v.get(i) > 0.5 && q.get(i) < 50) n++;
                            }
                        }
                    }
                    return n;
                },
                cfg -> {
                    long n = 0;
                    try (ParquetReader<Group> rd = groupReader(cfg.file)) {
                        Group g;
                        while ((g = rd.read()) != null) {
                            if (g.getFieldRepetitionCount("value") > 0 && g.getFieldRepetitionCount("qty") > 0
                                    && g.getDouble("value", 0) > 0.5 && g.getInteger("qty", 0) < 50) n++;
                        }
                    }
                    return n;
                }));

        w.add(new Workload("agg-sum-avg", "SUM and AVG over nullable columns",
                "SELECT SUM(value) AS s, AVG(qty) AS a FROM t",
                r -> col0(r, 0) + col0(r, 1),
                (cfg, a) -> {
                    double s = 0, q = 0;
                    long qn = 0;
                    try (ParquetBatchReader rd = scan(cfg, "value", "qty").open(a)) {
                        while (rd.next()) {
                            Float8Vector v = (Float8Vector) rd.root().getVector("value");
                            IntVector qv = (IntVector) rd.root().getVector("qty");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) {
                                if (!v.isNull(i)) s += v.get(i);
                                if (!qv.isNull(i)) { q += qv.get(i); qn++; }
                            }
                        }
                    }
                    return s + q / qn;
                },
                cfg -> {
                    double s = 0, q = 0;
                    long qn = 0;
                    try (ParquetReader<Group> rd = groupReader(cfg.file)) {
                        Group g;
                        while ((g = rd.read()) != null) {
                            if (g.getFieldRepetitionCount("value") > 0) s += g.getDouble("value", 0);
                            if (g.getFieldRepetitionCount("qty") > 0) { q += g.getInteger("qty", 0); qn++; }
                        }
                    }
                    return s + q / qn;
                }));

        w.add(new Workload("coalesce-sum", "COALESCE as a derived column inside SUM",
                "SELECT SUM(COALESCE(qty, 0)) AS s FROM t",
                r -> col0(r, 0),
                (cfg, a) -> {
                    double s = 0;
                    try (ParquetBatchReader rd = scan(cfg, "qty").open(a)) {
                        while (rd.next()) {
                            IntVector q = (IntVector) rd.root().getVector("qty");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) if (!q.isNull(i)) s += q.get(i);
                        }
                    }
                    return s;
                }, null));

        w.add(new Workload("like-native", "LIKE on a string column: native kernel mask",
                "SELECT COUNT(*) AS c FROM t WHERE name LIKE 'name-01%'",
                r -> col0(r, 0),
                (cfg, a) -> {
                    long n = 0;
                    byte[] prefix = "name-01".getBytes(StandardCharsets.UTF_8);
                    try (ParquetBatchReader rd = scan(cfg, "name").open(a)) {
                        while (rd.next()) {
                            VarCharVector v = (VarCharVector) rd.root().getVector("name");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) {
                                if (v.isNull(i)) continue;
                                byte[] b = v.get(i);
                                if (b.length >= prefix.length && Arrays.equals(b, 0, prefix.length, prefix, 0, prefix.length)) n++;
                            }
                        }
                    }
                    return n;
                }, null));

        w.add(new Workload("upper-eq", "UPPER(name) = 'NAME-0042': derived string column",
                "SELECT COUNT(*) AS c FROM t WHERE UPPER(name) = 'NAME-0042'",
                r -> col0(r, 0),
                (cfg, a) -> {
                    long n = 0;
                    byte[] want = "name-0042".getBytes(StandardCharsets.UTF_8);
                    try (ParquetBatchReader rd = scan(cfg, "name").open(a)) {
                        while (rd.next()) {
                            VarCharVector v = (VarCharVector) rd.root().getVector("name");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) {
                                if (v.isNull(i)) continue;
                                byte[] b = v.get(i);
                                if (b.length != want.length) continue;
                                boolean eq = true;
                                for (int k = 0; k < b.length && eq; k++) eq = Character.toLowerCase((char) b[k]) == want[k];
                                if (eq) n++;
                            }
                        }
                    }
                    return n;
                }, null));

        w.add(new Workload("extract-year", "EXTRACT(YEAR FROM day) = 2024: derived numeric column",
                "SELECT COUNT(*) AS c FROM t WHERE EXTRACT(YEAR FROM day) = 2024",
                r -> col0(r, 0),
                (cfg, a) -> {
                    long n = 0;
                    try (ParquetBatchReader rd = scan(cfg, "day").open(a)) {
                        while (rd.next()) {
                            DateDayVector d = (DateDayVector) rd.root().getVector("day");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) {
                                if (!d.isNull(i) && d.get(i) >= 19723 && d.get(i) <= 20088) n++;
                            }
                        }
                    }
                    return n;
                }, null));

        w.add(new Workload("group-numeric", "GROUP BY an int column, numeric engine",
                "SELECT qty, COUNT(*) AS c, SUM(value) AS s FROM t WHERE qty >= 0 GROUP BY qty",
                r -> r.getRowCount() * SPLIT + sumDouble(r, 1),
                (cfg, a) -> {
                    long[] cnt = new long[100];
                    try (ParquetBatchReader rd = scan(cfg, "qty").open(a)) {
                        while (rd.next()) {
                            IntVector q = (IntVector) rd.root().getVector("qty");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) if (!q.isNull(i)) cnt[q.get(i)]++;
                        }
                    }
                    long groups = 0, total = 0;
                    for (long x : cnt) if (x > 0) { groups++; total += x; }
                    return groups * SPLIT + total;
                }, null));

        w.add(new Workload("group-having", "GROUP BY with HAVING, ORDER BY and LIMIT: merged partials",
                "SELECT qty, COUNT(*) AS c FROM t WHERE qty >= 0 GROUP BY qty HAVING COUNT(*) > 20000 ORDER BY c DESC LIMIT 5",
                r -> sumDouble(r, 1),
                (cfg, a) -> {
                    long[] cnt = new long[100];
                    try (ParquetBatchReader rd = scan(cfg, "qty").open(a)) {
                        while (rd.next()) {
                            IntVector q = (IntVector) rd.root().getVector("qty");
                            for (int i = 0, m = rd.root().getRowCount(); i < m; i++) if (!q.isNull(i)) cnt[q.get(i)]++;
                        }
                    }
                    long[] sorted = cnt.clone();
                    Arrays.sort(sorted);
                    double s = 0;
                    for (int i = 0; i < 5; i++) if (sorted[sorted.length - 1 - i] > 20000) s += sorted[sorted.length - 1 - i];
                    return s;
                }, null));

        w.add(new Workload("group-string", "GROUP BY a string column: folded in Java",
                "SELECT name, COUNT(*) AS c FROM t GROUP BY name",
                r -> r.getRowCount() * SPLIT + sumDouble(r, 1),
                (cfg, a) -> {
                    Map<String, long[]> m = new HashMap<>();
                    try (ParquetBatchReader rd = scan(cfg, "name").open(a)) {
                        while (rd.next()) {
                            VarCharVector v = (VarCharVector) rd.root().getVector("name");
                            for (int i = 0, n = rd.root().getRowCount(); i < n; i++) {
                                String k = v.isNull(i) ? null : new String(v.get(i), StandardCharsets.UTF_8);
                                m.computeIfAbsent(k, x -> new long[1])[0]++;
                            }
                        }
                    }
                    long total = 0;
                    for (long[] x : m.values()) total += x[0];
                    return m.size() * SPLIT + total;
                }, null));

        w.add(new Workload("count-distinct", "COUNT(DISTINCT name): folded in Java",
                "SELECT COUNT(DISTINCT name) AS d FROM t",
                r -> col0(r, 0),
                (cfg, a) -> {
                    Set<String> s = new HashSet<>();
                    try (ParquetBatchReader rd = scan(cfg, "name").open(a)) {
                        while (rd.next()) {
                            VarCharVector v = (VarCharVector) rd.root().getVector("name");
                            for (int i = 0, n = rd.root().getRowCount(); i < n; i++) {
                                if (!v.isNull(i)) s.add(new String(v.get(i), StandardCharsets.UTF_8));
                            }
                        }
                    }
                    return s.size();
                }, null));

        w.add(new Workload("top-100", "ORDER BY value DESC LIMIT 100: bounded top-N",
                "SELECT id, value FROM t ORDER BY value DESC LIMIT 100",
                r -> sumDouble(r, 1),
                (cfg, a) -> {
                    PriorityQueue<Double> heap = new PriorityQueue<>(Comparator.naturalOrder());
                    try (ParquetBatchReader rd = scan(cfg, "value").open(a)) {
                        while (rd.next()) {
                            Float8Vector v = (Float8Vector) rd.root().getVector("value");
                            for (int i = 0, n = rd.root().getRowCount(); i < n; i++) {
                                if (v.isNull(i)) continue;
                                double x = v.get(i);
                                if (heap.size() < 100) heap.add(x);
                                else if (x > heap.peek()) { heap.poll(); heap.add(x); }
                            }
                        }
                    }
                    double s = 0;
                    for (double x : heap) s += x;
                    return s;
                }, null));
        w.add(new Workload("top-100-where", "ORDER BY value DESC LIMIT 100 with a WHERE: the engine filters, the heap picks",
                "SELECT id, value FROM t WHERE qty < 50 ORDER BY value DESC LIMIT 100",
                r -> sumDouble(r, 1),
                (cfg, a) -> {
                    PriorityQueue<Double> heap = new PriorityQueue<>(Comparator.naturalOrder());
                    try (ParquetBatchReader rd = scan(cfg, "value", "qty").open(a)) {
                        while (rd.next()) {
                            Float8Vector v = (Float8Vector) rd.root().getVector("value");
                            IntVector q = (IntVector) rd.root().getVector("qty");
                            for (int i = 0, n = rd.root().getRowCount(); i < n; i++) {
                                if (v.isNull(i) || q.isNull(i) || q.get(i) >= 50) continue;
                                double x = v.get(i);
                                if (heap.size() < 100) heap.add(x);
                                else if (x > heap.peek()) { heap.poll(); heap.add(x); }
                            }
                        }
                    }
                    double s = 0;
                    for (double x : heap) s += x;
                    return s;
                }, null));
        return w;
    }

    // ------------------------------------------------------------------ answer helpers

    private static double col0(VectorSchemaRoot r, int c) {
        FieldVector v = r.getFieldVectors().get(c);
        return v.isNull(0) ? 0 : ((Float8Vector) v).get(0);
    }

    private static double sumLong(VectorSchemaRoot r, int c) {
        BigIntVector v = (BigIntVector) r.getFieldVectors().get(c);
        double s = 0;
        for (int i = 0, n = r.getRowCount(); i < n; i++) if (!v.isNull(i)) s += v.get(i);
        return s;
    }

    private static double sumLong(VectorSchemaRoot r, String c) {
        BigIntVector v = (BigIntVector) r.getVector(c);
        double s = 0;
        for (int i = 0, n = r.getRowCount(); i < n; i++) if (!v.isNull(i)) s += v.get(i);
        return s;
    }

    private static double sumDouble(VectorSchemaRoot r, int c) {
        Float8Vector v = (Float8Vector) r.getFieldVectors().get(c);
        double s = 0;
        for (int i = 0, n = r.getRowCount(); i < n; i++) if (!v.isNull(i)) s += v.get(i);
        return s;
    }

    private static double sumDouble(VectorSchemaRoot r, String c) {
        Float8Vector v = (Float8Vector) r.getVector(c);
        double s = 0;
        for (int i = 0, n = r.getRowCount(); i < n; i++) if (!v.isNull(i)) s += v.get(i);
        return s;
    }

    private static ParquetReader<Group> groupReader(Path file) throws IOException {
        LocalInputFile input = new LocalInputFile(file.toAbsolutePath());
        return new ParquetReader.Builder<Group>(input) {
            @Override
            protected ReadSupport<Group> getReadSupport() {
                return new GroupReadSupport();
            }
        }.build();
    }

    // ------------------------------------------------------------------ running

    private record Row(String name, String note, double answer, boolean agree, boolean leak, double planMs,
                       BenchStats.Result sql, BenchStats.Result direct, BenchStats.Result raw, String explainMode, String pruning) { }

    private static boolean close(double a, double b) {
        return Math.abs(a - b) <= 1e-9 * Math.max(1.0, Math.max(Math.abs(a), Math.abs(b)));
    }

    private static Row run(Cfg c, Workload w) throws Exception {
        ParquetSqlOptions opt = options(c);
        BenchStats.Result sql, direct = null, raw = null;
        double planMs;
        boolean agree = true, leak = false;
        String mode, pruning = "";
        double answer;

        // planning cost, on its own
        double[] pm = new double[5];
        for (int i = 0; i < pm.length; i++) {
            long t0 = System.nanoTime();
            try (ParquetSqlPlan p = ParquetSql.plan(c.file, w.sql(), opt)) {
                pm[i] = (System.nanoTime() - t0) / 1e6;
            }
        }
        Arrays.sort(pm);
        planMs = pm[pm.length / 2];

        RootAllocator alloc = new RootAllocator();
        try (ParquetSqlPlan plan = ParquetSql.plan(c.file, w.sql(), opt)) {
            String ex = plan.explain();
            mode = ex.lines().filter(l -> l.startsWith("mode:")).findFirst().orElse("mode: ?").replace("mode: ", "");
            if (c.explain) System.out.println("--- " + w.name() + "\n" + ex);
            if (plan.isStreamable()) {
                // a projection or filter is consumed batch by batch, as an application would: the answers are additive over batches
                sql = BenchStats.measure(() -> {
                    double[] acc = {0};
                    plan.stream(alloc, root -> acc[0] += w.answer().applyAsDouble(root));
                    return acc[0];
                }, c.warmup, c.reps, c.minRepMs);
            } else {
                sql = BenchStats.measure(() -> {
                    try (VectorSchemaRoot r = plan.execute(alloc)) {
                        return w.answer().applyAsDouble(r);
                    }
                }, c.warmup, c.reps, c.minRepMs);
            }
            answer = sql.answer;
            ScanMetrics m = plan.lastMetrics();
            if (m != null) pruning = m.rowGroupsSkipped() + "/" + m.rowGroupsInFile() + " groups skipped";
        }
        try {
            alloc.close();
        } catch (IllegalStateException leaked) {
            leak = true;
        }

        RootAllocator a2 = new RootAllocator();
        direct = BenchStats.measure(() -> w.direct().run(c, a2), c.warmup, c.reps, c.minRepMs);
        agree &= close(answer, direct.answer);
        try {
            a2.close();
        } catch (IllegalStateException leaked) {
            leak = true;
        }

        if (c.raw && w.raw() != null) {
            raw = BenchStats.measure(() -> w.raw().run(c), 1, 3, 0);
            agree &= close(answer, raw.answer);
        }
        return new Row(w.name(), w.note(), answer, agree, leak, planMs, sql, direct, raw, mode, pruning);
    }

    private static void print(List<Row> rows) {
        System.out.println();
        System.out.printf(Locale.ROOT, "%-15s %10s %10s %9s %9s %9s %8s %8s %10s  %s%n", "workload", "sql ms", "direct ms", "sql/dir", "raw ms", "raw/sql",
                "plan ms", "ok", "alloc/call", "path");
        System.out.println("-".repeat(150));
        for (Row r : rows) {
            System.out.printf(Locale.ROOT, "%-15s %10.2f %10.2f %9.2f %9s %9s %8.1f %8s %10s  %s%s%n",
                    r.name(), r.sql().medianMs, r.direct().medianMs, r.sql().medianMs / Math.max(r.direct().medianMs, 1e-9),
                    r.raw() == null ? "-" : String.format(Locale.ROOT, "%.1f", r.raw().medianMs),
                    r.raw() == null ? "-" : String.format(Locale.ROOT, "%.2fx", r.raw().medianMs / Math.max(r.sql().medianMs, 1e-9)),
                    r.planMs(), r.agree() ? (r.leak() ? "LEAK" : "yes") : "MISMATCH", BenchStats.human(r.sql().allocBytesPerCall),
                    r.explainMode(), r.pruning().isEmpty() ? "" : "  [" + r.pruning() + "]");
        }
        System.out.println();
        System.out.println("sql/dir: the cost of the SQL layer over a hand-written loop on the same scan (1.0 = no overhead; below 1.0 means the");
        System.out.println("engine beat the hand loop). raw/sql: parquet-java Group reader time over ParquetSql time (the 3x to 5x claim, with SQL on top).");
        System.out.println("alloc/call: bytes allocated per call by the calling thread; scan worker threads are not included.");
        for (Row r : rows) {
            if (!r.agree()) System.out.printf(Locale.ROOT, "MISMATCH %s: sql=%s direct=%s raw=%s%n", r.name(), r.answer(), r.direct().answer, r.raw() == null ? "-" : r.raw().answer);
            if (r.leak()) System.out.println("LEAK in " + r.name() + ": the allocator was not empty after the workload");
        }
    }

    private static void csv(Cfg c, List<Row> rows) throws IOException {
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(Path.of(c.csv)))) {
            w.println("workload,rows,sql_ms,sql_min_ms,sql_max_ms,direct_ms,raw_ms,plan_ms,alloc_bytes_per_call,gc_ms_per_call,agree,leak,path");
            for (Row r : rows) {
                w.printf(Locale.ROOT, "%s,%d,%.4f,%.4f,%.4f,%.4f,%s,%.3f,%d,%.4f,%b,%b,\"%s\"%n", r.name(), c.rows, r.sql().medianMs, r.sql().minMs, r.sql().maxMs,
                        r.direct().medianMs, r.raw() == null ? "" : String.format(Locale.ROOT, "%.4f", r.raw().medianMs), r.planMs(), r.sql().allocBytesPerCall,
                        r.sql().gcMsPerCall, r.agree(), r.leak(), r.explainMode());
            }
        }
    }

    private static void profile(Cfg c, Workload w) throws Exception {
        System.out.println("Profiling '" + w.name() + "' for " + c.seconds + " s: " + w.sql());
        ParquetSqlOptions opt = options(c);
        try (RootAllocator alloc = new RootAllocator(); ParquetSqlPlan plan = ParquetSql.plan(c.file, w.sql(), opt)) {
            for (int i = 0; i < c.warmup; i++) {
                try (VectorSchemaRoot r = plan.execute(alloc)) {
                    w.answer().applyAsDouble(r);
                }
            }
            long end = System.nanoTime() + c.seconds * 1_000_000_000L;
            int calls = 0;
            double acc = 0;
            long a0 = BenchStats.allocatedBytes();
            try (StackSampler s = new StackSampler(2).start()) {
                while (System.nanoTime() < end) {
                    try (VectorSchemaRoot r = plan.execute(alloc)) {
                        acc += w.answer().applyAsDouble(r);
                    }
                    calls++;
                }
                System.out.println(s.report(25, "com.github.gbenroscience"));
            }
            System.out.printf(Locale.ROOT, "%d calls, %.1f ms per call, %s allocated per call by this thread (checksum %s)%n", calls,
                    c.seconds * 1000.0 / Math.max(calls, 1), BenchStats.human((BenchStats.allocatedBytes() - a0) / Math.max(calls, 1)), acc != 0);
        }
    }

    public static void main(String[] args) throws Exception {
        Cfg c = parse(args);
        Path dir = Files.createTempDirectory("parquet-sql-bench");
        c.file = dir.resolve("bench.parquet");
        int exit = 0;
        try {
            System.out.printf(Locale.ROOT, "Machine: %d cores, Java %s, max heap %d MiB%n", Runtime.getRuntime().availableProcessors(),
                    System.getProperty("java.version"), Runtime.getRuntime().maxMemory() >> 20);
            System.out.printf(Locale.ROOT, "Writing %,d rows (%s, %d MiB row groups)...%n", c.rows, c.codec, c.rowGroupMb);
            long t0 = System.nanoTime();
            BenchData.write(c.file, c.rows, (long) c.rowGroupMb << 20, c.codec);
            System.out.printf(Locale.ROOT, "  %s in %.1f s, %d row groups%n", BenchStats.human(Files.size(c.file)), (System.nanoTime() - t0) / 1e9,
                    ParquetFileInfo.read(c.file).rowGroups().size());
            Files.readAllBytes(c.file); // warm the page cache

            List<Workload> all = workloads(c);
            if (c.profile != null) {
                Workload pick = all.stream().filter(x -> x.name().contains(c.profile)).findFirst()
                        .orElseThrow(() -> new IllegalArgumentException("No workload matches '" + c.profile + "'"));
                profile(c, pick);
                return;
            }
            System.out.printf(Locale.ROOT, "Methodology: %d warm-up calls, %d reps of >= %.0f ms, median; raw baselines 1 + 3 reps.%n", c.warmup, c.reps, c.minRepMs);
            List<Row> rows = new ArrayList<>();
            for (Workload w : all) {
                if (c.only != null && !w.name().contains(c.only)) continue;
                System.out.print("  " + w.name() + " ... ");
                System.out.flush();
                Row r = run(c, w);
                System.out.printf(Locale.ROOT, "%.2f ms%n", r.sql().medianMs);
                rows.add(r);
                if (!r.agree() || r.leak()) exit = 1;
            }
            print(rows);
            if (c.csv != null) csv(c, rows);
            System.out.println("(checksum of everything: " + BenchStats.sink() + ")");
        } finally {
            if (!c.keep) {
                try (var walk = Files.walk(dir)) {
                    walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
                }
            } else {
                System.out.println("Kept " + c.file);
            }
        }
        if (exit != 0) System.exit(exit);
    }

    private ParquetSqlBenchmark() { }
}
