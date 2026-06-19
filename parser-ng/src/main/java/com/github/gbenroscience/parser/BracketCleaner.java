/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.gbenroscience.parser;

import com.github.gbenroscience.parser.methods.Method;
import com.github.gbenroscience.util.VariableManager;
import static com.github.gbenroscience.parser.Operator.*;
import static com.github.gbenroscience.parser.Variable.*;
import com.github.gbenroscience.util.FunctionManager;
import java.util.*;

/**
 * Removes excess/redundant brackets from a tokenized mathematical expression **in-place**.
 * 
 * Now fully restores your original safety checks:
 * - isVariableString(...)
 * - isAtOperator(...)          ← handles @(x,y) correctly
 * - Method.isMethodName(...) / FUNCTIONS set
 * - isUnaryPreOperator(...)
 * 
 * Also keeps all the improvements you liked:
 * - Safe for function calls like sin(x), log(...), @(x,y)
 * - Removes redundant nested brackets ((x)) → (x)
 * - Removes single-token brackets (x) → x when safe
 * - Removes empty brackets () when safe
 * - Handles deep nesting (((x))) in multiple passes
 * 
 * @author GBEMIRO
 */
public class BracketCleaner {


    // Fast lookup for known functions (covers Method.isMethodName)
    private static final List<String> FUNCTIONS = Arrays.asList(Method.getAllFunctions());

  
    /**
     * Removes excess brackets **in-place**. 
     * The list you pass in is modified directly — exactly like your original method.
     * @param scanner 
     */
    public static void removeExcessBrackets(List<String> scanner) {
        if (scanner == null || scanner.isEmpty()) {
            return;
        }

        boolean changed;
        do {
            changed = false;
            int[] pairMap = buildPairMap(scanner);

            for (int i = 0; i < scanner.size(); i++) {
                if (!"(".equals(scanner.get(i)) || pairMap[i] == -1) {
                    continue;
                }

                int open = i;
                int close = pairMap[i];

                // 1. Redundant nested brackets ((...)) → (...)   (always safe)
                if (open + 1 < scanner.size() && pairMap[open + 1] == close - 1) {
                    scanner.remove(close);
                    scanner.remove(open);
                    changed = true;
                    break;
                }

                // 2. Single-token brackets (x) → x   with ALL your original safety checks
                if (close == open + 2) {
                    boolean mustKeep = false;
                    if (open > 0) {
                        String prev = scanner.get(open - 1);
                        if (isVariableString(prev) ||
                            isAtOperator(prev) ||
                            FUNCTIONS.contains(prev) ||      // covers Method.isMethodName
                            isUnaryPreOperator(prev)) {
                            mustKeep = true;
                        }
                    }
                    if (!mustKeep) {
                        scanner.remove(close);
                        scanner.remove(open);
                        changed = true;
                        break;
                    }
                }

                // 3. Empty brackets () → remove only when safe
                if (close == open + 1) {
                    boolean mustKeep = false;
                    if (open > 0) {
                        String prev = scanner.get(open - 1);
                        if (isVariableString(prev) ||
                            isAtOperator(prev) ||
                            FUNCTIONS.contains(prev) ||
                            isUnaryPreOperator(prev)) {
                            mustKeep = true;
                        }
                    }
                    if (!mustKeep) {
                        scanner.remove(close);
                        scanner.remove(open);
                        changed = true;
                        break;
                    }
                }
            }
        } while (changed);   // handles deep nesting like (((x)))
    }

    private static int[] buildPairMap(List<String> tokens) {
        int n = tokens.size();
        int[] pairMap = new int[n];
        Arrays.fill(pairMap, -1);
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            String t = tokens.get(i);
            if ("(".equals(t)) {
                stack.push(i);
            } else if (")".equals(t) && !stack.isEmpty()) {
                int open = stack.pop();
                pairMap[open] = i;
                pairMap[i] = open;
            }
        }
        return pairMap;
    }
  
    public static void main(String[] args) {
         
        MathExpression mexie = new MathExpression("v=@(x,y)sin(3*log((((4*x-3*y),3)))-4);v(3,2)");
    
        // Example with @ operator
        List<String> scanner = mexie.getScanner();

        System.out.println("FUNCTIONS:    " + FunctionManager.FUNCTIONS);
        System.out.println("Raw scanner:    " + scanner);
        removeExcessBrackets(scanner);
        System.out.println("Cleaned scanner:" + scanner );
        System.out.println("Cleaned scanner:" + scanner );
        System.out.println("eval:" + mexie.solve() );
    }
}