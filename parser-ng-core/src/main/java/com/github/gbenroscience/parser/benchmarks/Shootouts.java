package com.github.gbenroscience.parser.benchmarks;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathScanner;
import java.util.List;

import java.util.Arrays;
import java.util.Scanner;

/**
 *
 * @author GBEMIRO
 */
public class Shootouts {

    public static double findAverage(double[] data) {
        if (data == null || data.length == 0) {
            return Double.NaN;
        }
        if (data.length <= 3) {
            // Too few points for meaningful outlier removal – return simple average
            double sum = 0.0;
            for (double v : data) {
                sum += v;
            }
            return sum / data.length;
        }

        // Work on a copy to avoid modifying the original array
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        int n = sorted.length;

        // More accurate quartile calculation with linear interpolation (Type 7 / "default" method)
        double q1 = percentile(sorted, 25.0);
        double q3 = percentile(sorted, 75.0);
        double iqr = q3 - q1;

        // Use standard 1.5 * IQR fences (Tukey method)
        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;

        // Count inliers and compute sum for average
        double sum = 0.0;
        int count = 0;
        for (double v : sorted) {
            if (v >= lowerBound && v <= upperBound) {
                sum += v;
                count++;
            }
        }

        // Fallback if all values were outliers (rare but possible with extreme noise)
        if (count == 0) {
            // Return overall average as safe fallback
            sum = 0.0;
            for (double v : data) {
                sum += v;
            }
            return sum / data.length;
        }

        return sum / count;
    }

// Helper: percentile with linear interpolation (matches R's default, Python's numpy.percentile, etc.)
    private static double percentile(double[] sorted, double p) {
        if (sorted.length == 0) {
            return Double.NaN;
        }
        if (p <= 0) {
            return sorted[0];
        }
        if (p >= 100) {
            return sorted[sorted.length - 1];
        }

        double index = (p / 100.0) * (sorted.length - 1);
        int lower = (int) Math.floor(index);
        double fraction = index - lower;

        if (lower + 1 < sorted.length) {
            return sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]);
        } else {
            return sorted[lower];
        }
    }

    public static List<String> scan(String expr) {
        return new MathScanner(expr).scanner();
    }

    public static void testGG(String expr, int n) {
        List<String> tokens = scan(expr);
        double val
                = GG.evaluate(tokens);
        double[] deltas = new double[n];

        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = GG.evaluate(tokens);
            deltas[i] = System.nanoTime() - deltas[i];
        }

        double avg = findAverage(deltas);
        System.out.println("GG-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void testGG2(String expr, int n) {
        List<String> tokens = scan(expr);
        GG2 g2 = new GG2();
        double val
                = g2.evaluate(tokens);
        double[] deltas = new double[n];

        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = g2.evaluate(tokens);
            deltas[i] = System.nanoTime() - deltas[i];
        }

        double avg = findAverage(deltas);
        System.out.println("GG2-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void testMC(String expr, int n) {
        List<String> tokens = scan(expr);

        double val
                = MC.evaluate(tokens);

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = MC.evaluate(tokens);
            deltas[i] = System.nanoTime() - deltas[i];
        }

        double avg = findAverage(deltas);
        System.out.println("MC-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void testTwG(String expr, int n) {
        List<String> tokens = scan(expr);

        double val = new TwG().evaluate(tokens);

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = new TwG().evaluate(tokens);
            deltas[i] = System.nanoTime() - deltas[i];
        }

        double avg = findAverage(deltas);
        System.out.println("TwG-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void testTwG2(String expr, int n) {
        List<String> tokens = scan(expr);

        double val = TwG2.evaluate(tokens);

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = TwG2.evaluate(tokens);
            deltas[i] = System.nanoTime() - deltas[i];
        }

        double avg = findAverage(deltas);
        System.out.println("TwG2-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void testTwGMBP(String expr, int n) {
        List<String> tokens = scan(expr);

        double val = TwGMBP.evaluate(tokens);

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = TwGMBP.evaluate(tokens);
            deltas[i] = System.nanoTime() - deltas[i];
        }
        double avg = findAverage(deltas);
        System.out.println("TwGMBP-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void testJavaNativeParser(String expr, int n) {
        TwG g = new TwG();
        double val = Math.pow(2, 3) + 4 % 2 - 5 - 6 - 7 * 8 + Maths.fact(5) + 2E-9 - 0.00002 + 70000 / Math.pow(32.34, 8) - 19 + g.permutation(9, 3) + g.combination(6, 5) + Math.pow(2, 2) + Math.pow(5, 3) - Math.pow(3, -1) / 2.53 + 3E-12;

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = Math.pow(2, 3) + 4 % 2 - 5 - 6 - 7 * 8 + Maths.fact(5) + 2E-9 - 0.00002 + 70000 / Math.pow(32.34, 8) - 19 + g.permutation(9, 3) + g.combination(6, 5) + Math.pow(2, 2) + Math.pow(5, 3) - Math.pow(3, -1) / 2.53 + 3E-12;
            deltas[i] = System.nanoTime() - deltas[i];
        }
        double avg = findAverage(deltas);
        System.out.println("Java-Native-Parser-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void benchmark(String expr, int n) {
        MathExpression m = new MathExpression(expr);

        String val = m.solve();

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = m.solve();
            deltas[i] = System.nanoTime() - deltas[i];
        }
        double avg = findAverage(deltas);
        System.out.println("MathExpression-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void benchmark(MathExpression expr, int n) {
        String val = expr.solve();

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            val = expr.solve();
            deltas[i] = System.nanoTime() - deltas[i];
        }
        double avg = findAverage(deltas);
        System.out.println("MathExpression-soln: " + val + ", " + (avg / 1000) + " microns");
    }

    public static void benchmark(Runnable task, int n) {

        double[] deltas = new double[n];
        double N = n;
        for (int i = 0; i < N; i++) {
            deltas[i] = System.nanoTime();
            task.run();
            deltas[i] = System.nanoTime() - deltas[i];
        }
        double avg = findAverage(deltas);
        System.out.println("Task average duration over " + n + " runs = " + (avg / 1000) + " microns");
    }

    /**
     * Measures and prints the memory used by a specific parser.
     */
    private static void profileMemory(String parserName, Runnable parserAction) {
        Runtime runtime = Runtime.getRuntime();

        // 1. Suggest the JVM clean up before we take our baseline measurement
        System.gc();

        // Give the GC a brief moment to actually run (optional but helps stability)
        try {
            Thread.sleep(10);
        } catch (InterruptedException e) {
        }

        // 2. Take the "Before" snapshot
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

        // 3. Run the parser benchmark
        parserAction.run();

        // 4. Take the "After" snapshot
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsedBytes = memoryAfter - memoryBefore;

        // Convert to Kilobytes for readability
        long memoryUsedKb = memoryUsedBytes / 1024;

        System.out.printf("%-15s Memory Used: %d KB (%,d bytes)%n", parserName, memoryUsedKb, memoryUsedBytes);
    }

    public static void main(String args[]) {

        //String s5 = "34/21+3";//
        String s5 = "2-12+2^3+4%2-5-6-7*8+5!+2E-9-0.00002+70000/32.34^8-19+9Р3+6Č5+2²+5³-3-¹/2.53+3E-12+2*--3";//
        System.out.println("EQUATION of the SHOOTOUT: "+s5);

        MathExpression m5 = new MathExpression(s5);
        System.out.println("True solution: " + m5.solve());

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.println("Enter n. If n is 0 or less, the process will be terminated");

            final int n = sc.nextInt();

            System.out.println("For " + n + " runs to warmup the JVM");
            if (n <= 0) {
                return;
            }
            profileMemory("TwG", () -> {
                // Put your loop or benchmark call here
                try {
                    testTwG(s5, n);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            profileMemory("TwG2", () -> {
                // Put your loop or benchmark call here
                try {
                    testTwG2(s5, n);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            profileMemory("TwGMBP", () -> {
                // Put your loop or benchmark call here
                try {
                    testTwGMBP(s5, n);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            profileMemory("GG", () -> {
                // Put your loop or benchmark call here
                try {
                    testGG(s5, n);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            profileMemory("GG2", () -> {
                // Put your loop or benchmark call here
                try {
                    testGG2(s5, n);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            profileMemory("MC", () -> {
                // Put your loop or benchmark call here
                try {
                    testMC(s5, n);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
            profileMemory("Java_Compiler_Parser", () -> {
                // Put your loop or benchmark call here
                try {
                    testJavaNativeParser(s5, n);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

        }

    }

}
