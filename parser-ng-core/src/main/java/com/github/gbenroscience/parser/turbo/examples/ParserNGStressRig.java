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
package com.github.gbenroscience.parser.turbo.examples;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.TurboEvaluatorFactory;
import java.util.concurrent.*; 
import java.util.logging.Level;
import java.util.logging.Logger;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;

public class ParserNGStressRig {
    private static final int THREADS = Runtime.getRuntime().availableProcessors();
    private static final int ITERATIONS_PER_THREAD = 1_000_000;
    private static final String TEST_EXPR_0 = "linear_sys(3,1,-2,4, 2,5)"; // Replace with complex Matrix ops
    private static final String TEST_EXPR_1 = "linear_sys(3,1,-2,4, 2,5,-8,6, 4,3,12,-18)"; // Replace with complex Matrix ops
    private static final String TEST_EXPR_2= "linear_sys(3,1,-2,4,5,-8, 2,5,-8,6,12,23, ,23,4,3,12,8,14, 1,3,2,5,4,7, 4,19,12,-3,18,50)"; // Replace with complex Matrix ops

    private static final String TEST_EXPR_X = "linear_sys(" +
    "10, 1, 0, 2, 3, 0, 1, 0, 4, 1, 50, " +  // Row 1
    "1, 12, 1, 0, 0, 4, 1, 2, 0, 3, 60, " +  // Row 2
    "0, 1, 15, 2, 1, 0, 3, 1, 4, 2, 70, " +  // Row 3
    "2, 0, 2, 20, 1, 5, 0, 1, 2, 1, 80, " +  // Row 4
    "3, 0, 1, 1, 25, 2, 1, 0, 3, 4, 90, " +  // Row 5
    "0, 4, 0, 5, 2, 30, 1, 2, 1, 0, 100, " + // Row 6
    "1, 1, 3, 0, 1, 1, 35, 4, 2, 1, 110, " + // Row 7
    "0, 2, 1, 1, 0, 2, 4, 40, 1, 3, 120, " + // Row 8
    "4, 0, 4, 2, 3, 1, 2, 1, 45, 2, 130, " + // Row 9
    "1, 3, 2, 1, 4, 0, 1, 3, 2, 50, 140"  + // Row 10
")";
       private static final String TEST_EXPR = TEST_EXPR_0;
    public static void main(String args[]) throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(THREADS);
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch endGate = new CountDownLatch(THREADS);
        
         
        System.out.printf("Starting Stress Test: %d threads, %d ops each...%n", THREADS, ITERATIONS_PER_THREAD);
        TurboExpressionEvaluator compiler = TurboEvaluatorFactory.getCompiler(new MathExpression(TEST_EXPR));
        for (int t = 0; t < THREADS; t++) {
            final int threadIdx = t;
            executor.submit(() -> {
                try {
                    // Warm up inside the thread to hydrate ThreadLocal
                    FastCompositeExpression fce = compiler.compile();
                    startGate.await(); 

                    double emptyFrame[] = {};
                    long start = System.nanoTime();
                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        fce.applyMatrix(emptyFrame);
                        
                        // Sampling for Jitter/Tail Latency
                        if (i % 1000 == 0 && threadIdx < 1000) {
                             // Record internal latency samples here
                        }
                    }
                    long end = System.nanoTime();
                    
                    double durationSeconds = (end - start) / 1_000_000_000.0;
                    double opsPerSec = ITERATIONS_PER_THREAD / durationSeconds;
                    System.out.printf("Thread %d: %.2f M-ops/sec%n", threadIdx, opsPerSec / 1_000_000);
                    
                } catch (Exception e) {
                    e.printStackTrace();
                } catch (Throwable ex) {
                    Logger.getLogger(ParserNGStressRig.class.getName()).log(Level.SEVERE, null, ex);
                } finally {
                    endGate.countDown();
                }
            });
        }

        long globalStart = System.nanoTime();
        startGate.countDown(); // Release the hounds
        endGate.await();
        long globalEnd = System.nanoTime();

        double totalDuration = (globalEnd - globalStart) / 1_000_000_000.0;
        long totalOps = (long) THREADS * ITERATIONS_PER_THREAD;
        System.out.println("--------------------------------------------------");
        System.out.printf("TOTAL THROUGHPUT: %.2f Million ops/sec%n", (totalOps / totalDuration) / 1_000_000);
        System.out.printf("AVG LATENCY: %.2f ns%n", (globalEnd - globalStart) / (double) totalOps);
        
        executor.shutdown();
    }
}