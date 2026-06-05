/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser.turbo;

/**
 *
 * @author GBEMIRO
 */
public class QuickTime {

    /**
     * Functional interface for the method to be timed.
     */
    @FunctionalInterface
    public interface Timable {

        void run();
    }

    /**
     * Times a single execution of a method in nanoseconds. Note: Single runs
     * are susceptible to JIT "cold" overhead.
     *
     * * @param task The logic to time.
     * @return Execution time in nanoseconds.
     */
    public static long timeNano(Timable task) {
        long start = System.nanoTime();
        task.run();
        return System.nanoTime() - start;
    }

    /**
     * Times a method over multiple iterations and returns the average. This is
     * better for benchmarking ParserNG logic.
     * @param label 
     * @param iterations Number of times to run the task.
     * @param warmups Number of times to run the task.
     * @param task The logic to time.
     * @return Average execution time in nanoseconds.
     */
    public static double benchmarkNano(String label,int warmups, int iterations,  Timable task) {
        if (iterations <= 0) {
            return 0;
        }

        // Warm up the JIT compiler first (optional but recommended)
        for (int i = 0; i < warmups; i++) {
            task.run();
        }

        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            task.run();
        }
        long totalTime = System.nanoTime() - start;

        double time = (double) totalTime / iterations;
        System.out.printf("[%s] Execution Time: %,f ns%n", label, time);
        return time;
    }

    /**
     * Prints a formatted report of the execution time.
     *
     * @param label
     * @param task
     */
    public static void report(String label, Timable task) {
        long time = timeNano(task);
        System.out.printf("[%s] Execution Time: %,d ns%n", label, time);
    }
}
