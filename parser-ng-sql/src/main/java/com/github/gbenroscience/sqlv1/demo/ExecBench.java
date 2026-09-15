package com.github.gbenroscience.sqlv1.demo;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.pojo.*;
import java.util.*;

public class ExecBench {
    static VectorSchemaRoot makeRoot(RootAllocator allocator, int n) {
        Field xField = new Field("x", FieldType.nullable(new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)), null);
        Field yField = new Field("y", FieldType.nullable(new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)), null);
        Float8Vector x = (Float8Vector) xField.createVector(allocator);
        Float8Vector y = (Float8Vector) yField.createVector(allocator);
        x.allocateNew(n); y.allocateNew(n);
        for (int i = 0; i < n; i++) { x.set(i, Math.sin(i) * 500 + i * 0.01); y.set(i, Math.cos(i) * 300 - i * 0.02); }
        x.setValueCount(n); y.setValueCount(n);
        return new VectorSchemaRoot(List.of(xField, yField), List.of(x, y), n);
    }

    interface Compiled extends AutoCloseable {
        VectorSchemaRoot execute(VectorSchemaRoot root);
        void close();
    }

    static double[] measure(Compiled q, VectorSchemaRoot root, int reps, int callsPerRep) {
        double[] us = new double[reps];
        for (int r = 0; r < reps; r++) {
            System.gc();
            try { Thread.sleep(50); } catch (InterruptedException ignored) {}
            long s = System.nanoTime();
            for (int i = 0; i < callsPerRep; i++) q.execute(root).close();
            long e = System.nanoTime();
            us[r] = (e - s) / 1000.0 / callsPerRep;
        }
        return us;
    }

    static void printStats(String label, double[] us, int batchRows) {
        double[] sorted = us.clone();
        Arrays.sort(sorted);
        double median = sorted[sorted.length / 2];
        System.out.printf("%-16s median=%.2f us  min=%.2f  max=%.2f  (%.1f M rows/sec at median)%n",
            label, median, sorted[0], sorted[sorted.length - 1], batchRows / median);
    }

    public static void main(String[] args) throws Throwable {
        final int BATCH = 100_000;
        final int WARMUP = 500;
        final int REPS = 9;
        final int CALLS_PER_REP = 800;
       // String sql = "SELECT x, y, sqrt(x*x+y*y) AS m FROM t";
        String sql = "SELECT x, y, sqrt(x*x+y*y) AS m FROM t WHERE NOT (x > 5) AND y < 250";

        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            VectorSchemaRoot root = makeRoot(allocator, BATCH);
 
            var qD = com.github.gbenroscience.sqlv1.ArrowQuery.compile(sql);

           
            Compiled cD = new Compiled() {
                public VectorSchemaRoot execute(VectorSchemaRoot r) { return qD.execute(r); }
                public void close() {}
            };

            for (int i = 0; i < WARMUP; i++) { cD.execute(root).close(); }

            double[] usD = new double[REPS];
            for (int r = 0; r < REPS; r++) {
                System.gc(); try { Thread.sleep(50); } catch (InterruptedException ignored) {}
                long s = System.nanoTime();  
                for (int i = 0; i < CALLS_PER_REP; i++) cD.execute(root).close();
                usD[r] = (System.nanoTime() - s) / 1000.0 / CALLS_PER_REP;
            }

            System.out.println("=== " + REPS + " interleaved reps x " + CALLS_PER_REP + " calls, " + BATCH + "-row batch ==="); 
            printStats("implD_fixed(orig)", usD, BATCH);  
            System.out.println();
            System.out.println("NOTE: run separately against implD_opt on the classpath swap below."); 
            qD.close();
            root.close();
        }
    }
}