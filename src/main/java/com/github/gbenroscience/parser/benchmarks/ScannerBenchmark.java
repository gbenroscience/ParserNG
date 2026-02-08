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

 
import com.github.gbenroscience.parser.CustomScanner; 
import com.github.gbenroscience.parser.Scanner;
import java.util.*;
/*
Sampel output gains:
Input Size: 1054 KB
--- Starting Benchmark ---

Scanner (Original) Time: 19085 ms
Scanner (Optimized) Time: 55 ms

Speedup Factor: 347.00x faster

So Scanner will replace Scanner in all parser code
*/
public class ScannerBenchmark {
    public static void main(String[] args) {
        // 1. Setup tokens (Operators, Keywords, etc.)
        String[] tokens = {"if", "else", "while", "return", "class", "public", "private", 
                           "==", "!=", ">=", "<=", "&&", "||", "++", "--", "+", "-", "*", "/", 
                           "=", "{", "}", "(", ")", "[", "]", ";", ",", ".", "String", "int"};

        // 2. Generate a Large Input (approx 1MB of text)
        StringBuilder sb = new StringBuilder();
        String sampleBody = "public class Test { int a = 10; if(a == 10 && true) { return a + 5; } } ";
        for (int i = 0; i < 15000; i++) {
            sb.append(sampleBody);
        }
        String largeInput = sb.toString();

        System.out.println("Input Size: " + (largeInput.length() / 1024) + " KB");
        System.out.println("--- Starting Benchmark ---\n");

        // Warm up the JVM (important for JIT compiler optimization)
        runScanner(largeInput, tokens);
        runOptimizedScanner(largeInput, tokens);

        // 3. Test Scanner (Original)
        long start1 = System.currentTimeMillis();
        runScanner(largeInput, tokens);
        long end1 = System.currentTimeMillis();
        System.out.println("Scanner (Original) Time: " + (end1 - start1) + " ms");

        // 4. Test Scanner (Optimized)
        long start2 = System.currentTimeMillis();
        runOptimizedScanner(largeInput, tokens);
        long end2 = System.currentTimeMillis();
        System.out.println("Scanner (Optimized) Time: " + (end2 - start2) + " ms");

        double speedup = (double) (end1 - start1) / (end2 - start2);
        System.out.println("\nSpeedup Factor: " + String.format("%.2f", speedup) + "x faster");
    }

    private static void runScanner(String input, String[] tokens) {
        new CustomScanner(input, true, tokens).scan();
    }

    private static void runOptimizedScanner(String input, String[] tokens) {
        new Scanner(input, true, tokens).scan();
    }
}