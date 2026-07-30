package com.github.gbenroscience.simdext.turbo.tools.bench;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import java.util.Scanner;
/**
 * 
 * Set as main class in pom and run OR:
 * 
 * FROM TERMINAL, do:
mvn exec:exec -Dexec.executable="java" -Dexec.args="--add-modules jdk.incubator.vector -classpath %classpath com.github.gbenroscience.simd.turbo.tools.bench.SIMDThrptBench"

 * 
 * 
 * @author oluwagbemirojiboye
 */
public class SIMDThrptBench {

    private static final int WARMUP_ITERATIONS = 40;
    private static final int MEASUREMENT_ITERATIONS = 150;

    public static void main(String[] args) throws Throwable {
        String rawExpr;

        // 1. Handle Dynamic Expression Entry
        if (args.length > 0 && !args[0].isBlank()) {
            rawExpr = args[0];
            System.out.println("Using expression from command line argument.");
        } else {
            System.out.println("No expression provided via CLI arguments.");
            System.out.println("Enter math expression (e.g., sin(sqrt(x^2+y^2)) ):");
            try (Scanner scanner = new Scanner(System.in)) {
                rawExpr = scanner.nextLine().trim();
            }
        }

        if (rawExpr.isBlank()) {
            System.err.println("Error: Expression cannot be empty.");
            return;
        }

        System.out.println("Compiling Expression: " + rawExpr);
        MathExpression me = new MathExpression(rawExpr);
        
        // Compile using the ParserNG Turbo Engine
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator;
        try {
            evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) 
                    new SIMDVectorTurboEvaluator(me).compile();
        } catch (Exception e) {
            System.err.println("Compilation failed. Ensure syntax and function support match ParserNG criteria.");
            e.printStackTrace();
            return;
        }

        // 2. Dynamically extract the variable count from the compiled expression
        // Assuming standard ParserNG descriptor inspection, or fallback to your explicit formula context
        int varCount = me.getVariables().getKey().length; 
        if (varCount == 0) {
            System.out.println("No variables found. Defaulting to 1 dummy variable slot to satisfy execution context.");
            varCount = 1;
        }
        
        System.out.println("Detected variables: " + me.getVariables() + " (" + varCount + " variables)");

        // 3. Setup bulk layout bounds
        int totalElements = 1000017; // 17 trailing to explicitly hit lane remainders
        double[][] inputs = new double[varCount][totalElements];
        double[] outputVector = new double[totalElements];

        // Seed continuous values into array primitives
        for (int v = 0; v < varCount; v++) {
            for (int i = 0; i < totalElements; i++) {
                inputs[v][i] = 1.0 + (i * 0.01) + (v * 0.5); 
            }
        }

        System.out.println("Memory allocation complete. Array elements per variable: " + totalElements);

        // 4. Benchmark Sequential Core Execution Loop
        System.out.println("\n--- Sequential SIMD Vector Engine (applyBulk) ---");
        runBenchmark(evaluator, inputs, outputVector, false, totalElements);

        // 5. Benchmark Parallel Fork-Join Execution Loop
        System.out.println("\n--- Parallel SIMD Multi-Core Engine (applyBulkParallel) ---");
        runBenchmark(evaluator, inputs, outputVector, true, totalElements);
    }

    private static void runBenchmark(
            SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator, 
            double[][] inputs, 
            double[] outputVector, 
            boolean parallel, 
            int totalElements) {

        System.out.print("Warming up JVM JIT compiler layers... ");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            if (parallel) {
                evaluator.applyBulkParallel(inputs, outputVector);
            } else {
                evaluator.applyBulk(inputs, outputVector);
            }
        }
        System.out.println("Stable.");

        System.out.print("Collecting performance throughput analytics... ");
        long startTime = System.nanoTime();
        
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            if (parallel) {
                evaluator.applyBulkParallel(inputs, outputVector);
            } else {
                evaluator.applyBulk(inputs, outputVector);
            }
        }
        
        long endTime = System.nanoTime();
        System.out.println("Done.");

        // Calculate performance scaling parameters
        double durationSeconds = (endTime - startTime) / 1_000_000_000.0;
        double iterationAverageMs = (durationSeconds * 1000.0) / MEASUREMENT_ITERATIONS;
        
        long totalCalculationsations = (long) MEASUREMENT_ITERATIONS * totalElements;
        double throughputOpsPerSec = totalCalculationsations / durationSeconds;
        double framesProcessedPerSecond = MEASUREMENT_ITERATIONS / durationSeconds;

        System.out.printf("  Average evaluation latency: %.3f ms per bulk payload%n", iterationAverageMs);
        System.out.printf("  Execution frequency       : %.2f full batches/sec%n", framesProcessedPerSecond);
        System.out.printf("  Total element throughput  : %,.0f element evaluations/sec (%.2fM ops/sec)%n", 
                throughputOpsPerSec, throughputOpsPerSec / 1_000_000.0);
    }
}