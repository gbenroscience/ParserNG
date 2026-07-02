/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
package com.github.gbenroscience.parser;

import java.util.*;
import java.util.function.Predicate;

/**
 * A highly optimized, customizable Lexical Analyzer (Scanner) for ParserNG.
 * <p>
 * This scanner breaks down mathematical expressions and source code strings into
 * distinct tokens using a longest-match (Max Munch) prefix strategy. It supports
 * static token dictionaries, dynamic identifier matching (via Predicates), and
 * optional whitespace stripping.
 *
 * @author GBEMIRO
 */
public class Scanner {

    private final String input;
    private final boolean includeTokensInOutput;
    private final boolean ignoreWhitespace;
    private final Map<Character, List<String>> tokensByFirstChar;
    private final Predicate<String> dynamicTokenMatcher;

    // =========================================================================
    // LEGACY CONSTRUCTORS (100% Backward Compatible)
    // =========================================================================

    public Scanner(String input, boolean includeTokensInOutput, String... splitterTokens) {
        this(new Builder(Objects.requireNonNull(input)).includeTokens(includeTokensInOutput).addTokens(splitterTokens));
    }

    public Scanner(String input, boolean includeTokensInOutput, String[] moreTokens, String... tokens) {
        this(new Builder(Objects.requireNonNull(input)).includeTokens(includeTokensInOutput).addTokens(moreTokens).addTokens(tokens));
    }

    public Scanner(String input, boolean includeTokensInOutput, String[] splitterTokens, String[] splitterTokens1, String... splitterTokens2) {
        this(new Builder(Objects.requireNonNull(input)).includeTokens(includeTokensInOutput)
                .addTokens(splitterTokens, splitterTokens1, splitterTokens2));
    }

    public Scanner(String input, boolean includeTokensInOutput, Predicate<String> dynamicTokenMatcher, String... splitterTokens) {
        this(new Builder(Objects.requireNonNull(input)).includeTokens(includeTokensInOutput)
                .withDynamicMatcher(dynamicTokenMatcher).addTokens(splitterTokens));
    }

    public Scanner(String input, boolean includeTokensInOutput, Predicate<String> dynamicTokenMatcher,
                   String[] moreTokens, String... tokens) {
        this(new Builder(Objects.requireNonNull(input)).includeTokens(includeTokensInOutput)
                .withDynamicMatcher(dynamicTokenMatcher).addTokens(moreTokens).addTokens(tokens));
    }

    public Scanner(String input, boolean includeTokensInOutput, Predicate<String> dynamicTokenMatcher,
                   String[] splitterTokens, String[] splitterTokens1, String... splitterTokens2) {
        this(new Builder(Objects.requireNonNull(input)).includeTokens(includeTokensInOutput)
                .withDynamicMatcher(dynamicTokenMatcher)
                .addTokens(splitterTokens, splitterTokens1, splitterTokens2));
    }

    // =========================================================================
    // CORE BUILDER INTEGRATION
    // =========================================================================
    private Scanner(Builder builder) {
        this.input = builder.input;
        this.includeTokensInOutput = builder.includeTokensInOutput;
        this.ignoreWhitespace = builder.ignoreWhitespace;
        this.dynamicTokenMatcher = builder.dynamicTokenMatcher;

        Map<Character, List<String>> map = new HashMap<>();
        for (String token : builder.allTokens) {
            if (token != null && !token.isEmpty()) {
                map.computeIfAbsent(token.charAt(0), k -> new ArrayList<>()).add(token);
            }
        }

        // Longest match first
        for (List<String> list : map.values()) {
            list.sort((a, b) -> Integer.compare(b.length(), a.length()));
        }

        this.tokensByFirstChar = Collections.unmodifiableMap(map);
    }

    // =========================================================================
    // SCANNER LOGIC
    // =========================================================================

    /**
     * Parses the input string into tokens and literals.
     * @return A sequential List of parsed string segments.
     */
    public List<String> scan() {
        List<String> output = new ArrayList<>();
        int cursor = 0;
        int literalStart = 0;
        int length = input.length();

        while (cursor < length) {
            char currentChar = input.charAt(cursor);

            // 1. Whitespace Fast-Forwarding
            if (ignoreWhitespace && Character.isWhitespace(currentChar)) {
                flushLiteral(output, literalStart, cursor);

                while (cursor < length && Character.isWhitespace(input.charAt(cursor))) {
                    cursor++;
                }

                literalStart = cursor;
                continue;
            }

            // 2. Static Token Matching
            List<String> candidates = tokensByFirstChar.getOrDefault(currentChar, Collections.emptyList());
            boolean matched = false;

            for (String token : candidates) {
                int tokenLen = token.length();
                if (tokenLen <= length - cursor && input.regionMatches(cursor, token, 0, tokenLen)) {
                    flushLiteral(output, literalStart, cursor);
                    if (includeTokensInOutput) {
                        output.add(token);
                    }
                    cursor += tokenLen;
                    literalStart = cursor;
                    matched = true;
                    break;
                }
            }

            // 3. Dynamic Identifier Matching
            if (!matched && Character.isJavaIdentifierStart(currentChar)) {
                int end = cursor + 1;
                while (end < length && Character.isJavaIdentifierPart(input.charAt(end))) {
                    end++;
                }
                String potentialWord = input.substring(cursor, end);
                if (dynamicTokenMatcher.test(potentialWord)) {
                    flushLiteral(output, literalStart, cursor);
                    output.add(potentialWord);
                    cursor = end;
                    literalStart = cursor;
                    continue;
                }
            }

            if (!matched) {
                cursor++;
            }
        }

        flushLiteral(output, literalStart, length);
        return output;
    }

    private void flushLiteral(List<String> output, int start, int end) {
        if (end > start) {
            String literal = input.substring(start, end);
            if (!ignoreWhitespace || !literal.trim().isEmpty()) {
                output.add(literal);
            }
        }
    }

    // =========================================================================
    // BUILDER
    // =========================================================================

    public static class Builder {
        private final String input;
        private boolean includeTokensInOutput = true;
        private boolean ignoreWhitespace = false;
        private Predicate<String> dynamicTokenMatcher = s -> false;
        private final List<String> allTokens = new ArrayList<>();

        public Builder(String input) {
            this.input = Objects.requireNonNull(input, "Input string cannot be null");
        }

        public Builder includeTokens(boolean include) {
            this.includeTokensInOutput = include;
            return this;
        }

        public Builder ignoreWhitespace(boolean ignore) {
            this.ignoreWhitespace = ignore;
            return this;
        }

        public Builder withDynamicMatcher(Predicate<String> matcher) {
            this.dynamicTokenMatcher = matcher != null ? matcher : s -> false;
            return this;
        }

        public Builder addTokens(String[]... tokenArrays) {
            for (String[] array : tokenArrays) {
                if (array != null) {
                    this.allTokens.addAll(Arrays.asList(array));
                }
            }
            return this;
        }

        public Scanner build() {
            return new Scanner(this);
        }
    }
    
    // =========================================================================
    // TESTING
    // =========================================================================

    public static void main(String[] args) {
        // Test expression with spaces and valid/invalid identifiers
        String testInput = "print( anon9 , _anon2 , $C )";
        String[] standardTokens = {"print", "(", ")", ","};

        // Matches 'anon' followed by digits OR starts with underscore
        Predicate<String> dynamicRules = word -> 
            (word.startsWith("anon") && word.substring(4).matches("\\d+")) || word.startsWith("_");

        // 1. Using Legacy Constructor (Will keep spaces as literals)
        Scanner scLegacy = new Scanner(testInput, true, dynamicRules, standardTokens);
        System.out.println("Legacy Output: " + scLegacy.scan());
        
        // 2. Using the New Builder (Ignoring whitespace)
        Scanner scBuilder = new Scanner.Builder(testInput)
                .includeTokens(true)
                .ignoreWhitespace(true)
                .withDynamicMatcher(dynamicRules)
                .addTokens(standardTokens)
                .build();
        
        System.out.println("Builder Output: " + scBuilder.scan());
    }
}