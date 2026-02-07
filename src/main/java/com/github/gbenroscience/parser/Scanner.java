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
package com.github.gbenroscience.parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Scanner {

    private final String input;
    private final boolean includeTokensInOutput;
    private final Map<Character, List<String>> tokensByFirstChar;
/**
     * Standard Constructor
     */
    public Scanner(String input, boolean includeTokensInOutput, String... splitterTokens) {
        this(input, includeTokensInOutput, combine(splitterTokens));
    }

    /**
     * Constructor for two arrays/varargs
     */
    public Scanner(String input, boolean includeTokensInOutput, String[] moreTokens, String... tokens) {
        this(input, includeTokensInOutput, combine(moreTokens, tokens));
    }

    /**
     * Constructor for three arrays/varargs
     */
    public Scanner(String input, boolean includeTokensInOutput, String[] splitterTokens, String[] splitterTokens1, String... splitterTokens2) {
        this(input, includeTokensInOutput, combine(splitterTokens, splitterTokens1, splitterTokens2));
    }

    // --- Private logic ---

    /**
     * Internal Master Constructor
     */
    private Scanner(String input, boolean includeTokensInOutput, List<String> allTokens) {
        this.input = input;
        this.includeTokensInOutput = includeTokensInOutput;

        Map<Character, List<String>> map = new HashMap<>();
        for (String token : allTokens) {
            if (token != null && !token.isEmpty()) {
                char first = token.charAt(0);
                if (!map.containsKey(first)) {
                    map.put(first, new ArrayList<String>());
                }
                map.get(first).add(token);
            }
        }

        // Sort each group once during initialization
        Comparator<String> lengthDesc = new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                return Integer.compare(b.length(), a.length());
            }
        };

        for (List<String> list : map.values()) {
            Collections.sort(list, lengthDesc);
        }

        this.tokensByFirstChar = Collections.unmodifiableMap(map);
    }
     private static List<String> combine(String[]... arrays) {
        List<String> combined = new ArrayList<>();
        for (String[] array : arrays) {
            if (array != null) {
                combined.addAll(Arrays.asList(array));
            }
        }
        return combined;
    }
  

    public List<String> scan() {
        List<String> output = new ArrayList<>();
        int cursor = 0;
        int literalStart = 0;
        int length = input.length();

        while (cursor < length) {
            char currentChar = input.charAt(cursor);
            List<String> candidates = tokensByFirstChar.get(currentChar);
            if (candidates == null) {
                candidates = Collections.emptyList();
            }

            boolean matched = false;
            for (String token : candidates) {
                int tokenLen = token.length();
                int remaining = length - cursor;

                if (tokenLen <= remaining
                        && input.regionMatches(cursor, token, 0, tokenLen)) {

                    // Add literal text before the token
                    if (cursor > literalStart) {
                        output.add(input.substring(literalStart, cursor));
                    }

                    // Add the token if requested
                    if (includeTokensInOutput) {
                        output.add(token);
                    }

                    // Advance past the token
                    cursor += tokenLen;
                    literalStart = cursor;
                    matched = true;
                    break; // Longest match wins
                }
            }

            // No match → advance one char (becomes part of literal)
            if (!matched) {
                cursor++;
            }
        }

        // Add any trailing literal text
        if (literalStart < length) {
            output.add(input.substring(literalStart, length));
        }

        return output;
    }

    public static void main(String[] args) {

        String in
                = "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+"
                + "(28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)+sinh(8)+zopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekm-zopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekm";
        int N = 1000;
        String[] tokens = new String[]{"sinh", "+", "-", ")", "(", "sin", "cos", "/", "zopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekm-zopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekmzopmkdmekdekekdmekdmekdmekdmekdmekdmekdmekdmekdmekdmekm"};
        CustomScanner cs = new CustomScanner(in, true, tokens);
        Scanner sc = new Scanner(in, true, tokens);
        long start = System.nanoTime();
        List<String> csOut = null;
        for (int i = 0; i < N; i++) {
            csOut = cs.scan();
        }
        long duration = (System.nanoTime() - start) / N;
        System.out.println("Old Scanner parse in>>> " + (duration) + " ns");

        System.out.println("Output>>>\n " + csOut);

        start = System.nanoTime();
        List<String> scOut = null;
        for (int i = 0; i < N; i++) {
            scOut = sc.scan();
        }
        duration = (System.nanoTime() - start) / N;
        System.out.println("New Scanner parse in>>> " + (duration) + " ns");

        System.out.println("Output>>>\n " + scOut);

    }

}
