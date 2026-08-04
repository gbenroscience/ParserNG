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
package com.github.gbenroscience.parser.benchmarks;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.QuickTime;
import com.github.gbenroscience.util.Serializer;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author GBEMIRO
 */
public class MeCopyVsConstructor {

    private static volatile boolean toggle = true;

    private static final String expr = "(38*x+29*sin(x)^2+3*cos(x^2)^2)^2.82";

    // 1. Replaced ArrayList with a thread-safe, non-blocking Queue
    public static final ConcurrentLinkedQueue<MathExpression> TRASH = new ConcurrentLinkedQueue<>();
    private static final AtomicInteger deleteCount = new AtomicInteger();
    private static final int warmups = 1000;
    private static final int iterations = 10000;
    
    private static boolean runSequentially = true;

    public MeCopyVsConstructor() {
    }

    
    public static void main(String[] args) {
        if(runSequentially){
            mainSeq(args);
        }else{
            mainParallel(args);
        }
    }
    public static void mainParallel(String[] args) {
        // 2. Use a CountDownLatch to cleanly track when all 3 producers finish
        CountDownLatch producersLatch = new CountDownLatch(3);

        final Thread consumer = new Thread(() -> {
            // 3. Consolidated loop: run while active OR while there's still trash to process
            while (toggle || !TRASH.isEmpty()) {
                MathExpression m = TRASH.poll(); // O(1) removal
                if (m != null) {
                    m.solve();
                    deleteCount.incrementAndGet();
                } else {
                    // Prevent 100% CPU spin when the queue is momentarily empty
                    Thread.yield();
                }
            }
        });
        consumer.start();

        // Producer 1: Constructor
        new Thread(() -> {
            QuickTime.benchmarkNano("MathExpression-Constructor", warmups, iterations, () -> {
                MathExpression m = new MathExpression(expr);
                TRASH.add(m);
            });
            producersLatch.countDown();
        }).start();

        // Producer 2: Copy
        new Thread(() -> {
            final MathExpression m = new MathExpression(expr);
            QuickTime.benchmarkNano("MathExpression-Copy", warmups, iterations, () -> {
                MathExpression me = m.copy();
                TRASH.add(me);
            });
            producersLatch.countDown();
        }).start();

        // Producer 3: Clone
        new Thread(() -> {
            final MathExpression m = new MathExpression(expr);
            QuickTime.benchmarkNano("MathExpression-Clone", warmups, iterations, () -> {
                try {
                    MathExpression me = m.clone();
                    TRASH.add(me);
                } catch (CloneNotSupportedException ex) {
                    Logger.getLogger(MeCopyVsConstructor.class.getName()).log(Level.SEVERE, null, ex);
                }
            });
            producersLatch.countDown();
        }).start();

        try {
            // 4. Main thread waits for all benchmarks to complete
            producersLatch.await();

            // 5. Signal the consumer to stop after it finishes draining the queue
            toggle = false;
            System.out.println("Benchmarks complete. Waiting for consumer to drain...");

            // Wait for consumer to finish processing the queue safely
            consumer.join();
        } catch (InterruptedException i) {
            Thread.currentThread().interrupt();
            System.err.println("Main thread interrupted!");
        }

        System.out.println("deleteCount: " + deleteCount.get());
    }

    public static void mainSeq(String[] args) {
        // Dedicated Consumer Thread
        final Thread consumer = new Thread(() -> {
            while (toggle || !TRASH.isEmpty()) {
                MathExpression m = TRASH.poll();
                if (m != null) {
                    m.solve();
                    deleteCount.incrementAndGet();
                } else {
                    Thread.yield();
                }
            }
        });
        consumer.start();

        // Run the phases sequentially
        runPhase("COLD JIT", 10, 100);
        runPhase("WARM JIT", 100, 1000);
        runPhase("HOT C2 JIT", 1000, 10000);

        toggle = false; // Kill consumer
    }

    private static void runPhase(String phaseName, int warmup, int iterations) {
        System.out.println("\n--- " + phaseName + " (Warmup: " + warmup + ", Iterations: " + iterations + ") ---");

        final String expr = "(38*x+29*sin(x)^2+3*cos(x^2)^2)^2.82";
        final MathExpression template = new MathExpression(expr);

        // 1. Constructor
        QuickTime.benchmarkNano("MathExpression-Constructor", warmup, iterations, () -> {
            TRASH.add(new MathExpression(expr));
        });
        cleanUp();

        // 2. Clone (Serialization)
        QuickTime.benchmarkNano("MathExpression-Clone", warmup, iterations, () -> {
            try {
                TRASH.add(Serializer.deepClone(template));
            } catch (Exception ex) {
            }
        });
        cleanUp();

        // 3. Old Copy
        QuickTime.benchmarkNano("MathExpression-Copy", warmup, iterations, () -> {
            TRASH.add(template.copy()); // Assuming you renamed the old one
        });
        cleanUp();
 
    }

    /**
     * Ensures the queue is empty and memory is clean before the next test
     * starts.
     */
    private static void cleanUp() {
        while (!TRASH.isEmpty()) {
            Thread.yield(); // Wait for consumer to finish the current batch
        }
        System.gc(); // Request garbage collection to ensure a clean slate
        try {
            Thread.sleep(200); // Let the JVM settle
        } catch (InterruptedException e) {
        }
    }
}
