package com.github.gbenroscience.sqlv1.demo;

import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.sqlv1.ArrowQuery;
import com.github.gbenroscience.sqlv1.ArrowSql;
import com.github.gbenroscience.util.ConsoleTable;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * An end-to-end tour of parser-ng-sql: unlike {@link SqlDemo} (which only
 * parses SQL text), every example here builds a real Apache Arrow
 * {@code VectorSchemaRoot}, hands it to {@link ArrowQuery}/{@link ArrowSql},
 * and prints the actual computed/filtered result -- i.e. parser-ng-sql driving
 * parser-ng-arrow's {@code ArrowExpressionEvaluators} against real data,
 * exactly as it would be used in an application.
 *
 * <h2>A note on how this file was produced</h2>
 * This class depends on {@code arrow-vector} and {@code parser-ng-arrow},
 * neither of which was reachable in the sandbox this module was written in (no
 * Maven Central access, no local Arrow jars -- see the module
 * {@code README.md}'s "Verification status"). It was written carefully against
 * the exact same Arrow API calls already used (and reviewed) inside
 * {@code ArrowQuery} itself -- {@code Float8Vector}'s constructor/
 * {@code allocateNew}/{@code setSafe}/{@code setValueCount}, the
 * {@code VectorSchemaRoot(Schema, List<FieldVector>, int)} constructor,
 * {@code RootAllocator} -- but could not be compiled or run there. Please run
 * it yourself (`mvn -pl parser-ng-sql -am test-compile exec:java
 * -Dexec.mainClass=com.github.gbenroscience.sqlv1.demo.ArrowSqlDemo`, or just
 * run its {@code main} from your IDE) before relying on it, same as the rest of
 * {@code ArrowQuery}/{@code ArrowSql}.
 *
 * @author GBEMIRO
 */
public final class ArrowSqlDemo {

    public static void main(String[] args) {
        try (BufferAllocator allocator = new RootAllocator()) {

            runOneShot("SELECT x,y,", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x,y, 3*x+y, erf(x) AS erfx FROM data where x-erf(x)>0 ORDER BY erfx");
            
            
            runOneShot("SELECT x,y,", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x,y, 3*x+y, sin(x) AS sinx FROM data where x-erf(x)>0 ORDER BY sinx");
            
            
            runOneShot("SELECT x,y,", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x,y, 3*x+y, sin(x) AS sinx FROM data where x-erf(x)>0 ORDER BY sinx LIMIT 4");

            // ---- SELECT list forms -------------------------------------------
            runOneShot("SELECT * -- passthrough every column", allocator,
                    () -> sampleXY(allocator),
                    "SELECT * FROM data");

            runOneShot("Column subset, no computation", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x FROM data");

            runOneShot("Computed column with alias, no filter", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data");

            runOneShot("Computed column AND filter -- write the same expression in WHERE", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data WHERE sqrt(x*x + y*y) > 60");

            System.out.println("=== WHERE referencing a SELECT-list alias (via WhereAliasResolver) ===");
            System.out.println("ArrowQuery now expands a WHERE-clause name that matches a SELECT-list");
            System.out.println("alias back into the expression that alias stands for, before compiling");
            System.out.println("WHERE -- so naming the alias directly works exactly like repeating its");
            System.out.println("expression (previous example). A real input column always wins over a");
            System.out.println("same-named alias, and chained aliases (one alias's expression naming");
            System.out.println("another) resolve transitively.");
            runOneShot("  -> now works", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data WHERE magnitude > 60");

            // ---- WHERE: AND / OR / NOT / BETWEEN / IN ------------------------
            runOneShot("WHERE ... AND ...", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y FROM data WHERE x > 20 AND y < 90");

            runOneShot("WHERE ... OR ...", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y FROM data WHERE x < 20 OR x > 70");

            runOneShot("WHERE NOT ...", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y FROM data WHERE NOT x > 50");

            runOneShot("WHERE ... BETWEEN ... AND ...", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y FROM data WHERE x BETWEEN 20 AND 60");

            runOneShot("WHERE ... IN (...)", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, y FROM data WHERE x IN (10, 50, 90)");

            // ---- IS [NOT] NULL, against a column that actually has nulls -----
            runOneShot("WHERE ... IS NULL", allocator,
                    () -> sampleReadingsWithNulls(allocator),
                    "SELECT reading FROM data WHERE reading IS NULL");

            runOneShot("WHERE ... IS NOT NULL", allocator,
                    () -> sampleReadingsWithNulls(allocator),
                    "SELECT reading FROM data WHERE reading IS NOT NULL");

            // ---- embedded boolean condition inside a projection ---------------
            runOneShot("if(condition, a, b) as a computed projection column", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, if(x > 50, 1, 0) AS high_flag FROM data");
            
            
            // ---- embedded boolean condition inside a projection ---------------
            runOneShot("Another if(condition, a, b) as a computed projection column", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, if(sin(x) > 0, tan(x), 0.2) AS tan_vs_a_fifth FROM data");
            
                 // ---- embedded boolean condition inside a projection ---------------
            runOneShot("With BETWEEN: if(condition, a, b) as a computed projection column", allocator,
                    () -> sampleXY(allocator),
                    "SELECT x, if(x BETWEEN 1 AND 50, 1, 0) AS binary FROM data");


            // ---- compile once, execute many times against different batches --
            System.out.println("=== Compile once, execute many: the same ArrowQuery run against two batches ===");
            try (ArrowQuery reusable = ArrowQuery.compile(
                    "SELECT x, y, x * y AS product FROM data WHERE x > 15")) {
                runCompiled("  -> batch 1", reusable, sampleXY(allocator));
                runCompiled("  -> batch 2 (different data, same compiled plan)", reusable,
                        sampleXYRange(allocator, 100, 190, 10));
            }

            // ---- explicit backend / NullPolicy configuration ------------------
            System.out.println("=== NullPolicy.IGNORE (default) vs NullPolicy.PROPAGATE on a computed column ===");
            try (ArrowQuery ignorePolicy = ArrowQuery.compile("SELECT reading * 2 AS doubled FROM data")
                    .withBackend(ArrowExecutionBackend.CPU_SIMD)) {
                runCompiled("  -> IGNORE (default): nulls treated as if they were absent from the computation",
                        ignorePolicy, sampleReadingsWithNulls(allocator));
            }
            try (ArrowQuery propagatePolicy = ArrowQuery.compile("SELECT reading * 2 AS doubled FROM data")
                    .withBackend(ArrowExecutionBackend.CPU_SIMD)
                    .withNullPolicy(NullPolicy.PROPAGATE)) {
                runCompiled("  -> PROPAGATE: a null input keeps the output null instead of being skipped",
                        propagatePolicy, sampleReadingsWithNulls(allocator));
            }
        }
    }

    // =========================================================================
    // execution helpers
    // =========================================================================
    private static void runOneShot(String title, BufferAllocator allocator,
            Supplier<VectorSchemaRoot> rootFactory, String sql) {
        System.out.println("=== " + title + " ===");
        System.out.println("SQL: " + sql);
        try (VectorSchemaRoot input = rootFactory.get()) {
            System.out.println("Input:");
            printRoot(input);
            try (VectorSchemaRoot result = ArrowSql.execute(input, sql)) {
                System.out.println("Result:");
                printRoot(result);
            }
        } catch (RuntimeException e) {
            System.out.println("ERROR: " + e);
        }
        System.out.println();
    }

    private static void runCompiled(String title, ArrowQuery query, VectorSchemaRoot input) {
        System.out.println(title);
        try (VectorSchemaRoot in = input) {
            System.out.println("Input:");
            printRoot(in);
            try (VectorSchemaRoot result = query.execute(in)) {
                System.out.println("Result:");
                printRoot(result);
            }
        } catch (RuntimeException e) {
            System.out.println("ERROR: " + e);
        }
        System.out.println();
    }

    // =========================================================================
    // sample data
    // =========================================================================
    private static VectorSchemaRoot sampleXY(BufferAllocator allocator) {
        double[] xs = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
        double[] ys = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
        return root(col(allocator, "x", xs, null), col(allocator, "y", ys, null));
    }

    /**
     * Builds a fresh {@code x}/{@code y} batch of {@code count} rows, {@code x}
     * starting at {@code startX} and stepping by {@code stepX};
     * {@code y = x / 10} -- used to demonstrate the same compiled
     * {@link ArrowQuery} running against a second, differently-sized batch.
     */
    private static VectorSchemaRoot sampleXYRange(BufferAllocator allocator, double startX, double endX, double stepX) {
        List<Double> xList = new ArrayList<>();
        for (double x = startX; x <= endX + 1e-9; x += stepX) {
            xList.add(x);
        }
        double[] xs = new double[xList.size()];
        double[] ys = new double[xList.size()];
        for (int i = 0; i < xs.length; i++) {
            xs[i] = xList.get(i);
            ys[i] = xs[i] / 10.0;
        }
        return root(col(allocator, "x", xs, null), col(allocator, "y", ys, null));
    }

    private static VectorSchemaRoot sampleReadingsWithNulls(BufferAllocator allocator) {
        double[] values = {12.5, 0, 7.25, 0, 0, 42.0};
        boolean[] isNull = {false, true, false, true, true, false};
        return root(col(allocator, "reading", values, isNull));
    }

    private static Float8Vector col(BufferAllocator allocator, String name, double[] values, boolean[] nullMask) {
        Float8Vector v = new Float8Vector(name, allocator);
        v.allocateNew(values.length);
        for (int i = 0; i < values.length; i++) {
            if (nullMask != null && nullMask[i]) {
                v.setNull(i);
            } else {
                v.setSafe(i, values[i]);
            }
        }
        v.setValueCount(values.length);
        return v;
    }

    private static VectorSchemaRoot root(Float8Vector... columns) {
        List<Field> fields = new ArrayList<>();
        List<FieldVector> vectors = new ArrayList<>();
        for (Float8Vector v : columns) {
            fields.add(v.getField());
            vectors.add(v);
        }
        return new VectorSchemaRoot(new Schema(fields), vectors, columns[0].getValueCount());
    }

    // =========================================================================
    // printing
    // =========================================================================
    private static void printRoot(VectorSchemaRoot root) {
        List<FieldVector> vectors = root.getFieldVectors();
        int rowCount = root.getRowCount();
 
        String[] headers = new String[vectors.size()];
        int i = 0;
        for (FieldVector v : vectors) {
            headers[i++] = v.getName();
        }
        String[] data[] = new String[rowCount][headers.length];
        for (int r = 0; r < rowCount; r++) { 

            i = 0;
            for (FieldVector v : vectors) {
                String cell;
                if (v.isNull(r)) {
                    cell = "NULL"; 
                } else if (v instanceof Float8Vector f8) {
                    cell = String.format("%.4f", f8.get(r));
                } else {
                    cell = String.valueOf(v.getObject(r));
                }
                data[r][i++] = cell; 
            } 
        }
        new ConsoleTable("", headers, data).display();
        System.out.println("(" + rowCount + " row" + (rowCount == 1 ? "" : "s") + ")");
    }

    private ArrowSqlDemo() {
    }
}