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
 * <p>
 * <b>Match ordering:</b> by default, static tokens are tried before the dynamic
 * identifier predicate ("static-first"). This can be flipped to "dynamic-first"
 * via {@link Builder#matchDynamicFirst(boolean)}, which mirrors how many real
 * lexers treat keywords as a special case of identifiers rather than the other
 * way around. Note that in dynamic-first mode, a predicate that matches a string
 * equal to one of your static tokens will shadow that token — this is expected,
 * and is the caller's responsibility to account for when writing the predicate.
 *
 * @author GBEMIRO
 */
public class Scanner {

    private final String input;
    private final boolean includeTokensInOutput;
    private final boolean ignoreWhitespace;
    private final boolean dynamicMatchFirst;
    private final Map<Character, List<String>> tokensByFirstChar;
    private final Predicate<String> dynamicTokenMatcher;
    private final HashSet<Character> extraIdentifierParts;

    // Lazily computed, cached since the scanner is fully immutable post-construction.
    private List<String> cachedResult;

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
        this.dynamicMatchFirst = builder.dynamicMatchFirst;
        this.extraIdentifierParts = builder.extraIdentifierParts.isEmpty()
                ? new HashSet<>()
                : new HashSet<>(builder.extraIdentifierParts);

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
     * The result is computed once and cached, since a Scanner instance is immutable.
     *
     * @return A sequential List of parsed string segments.
     */
    public List<String> scan() {
        if (cachedResult != null) {
            return cachedResult;
        }

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

            // Precompute the identifier-shaped run starting here, if any.
            // Used both for dynamic matching and for guarding static matches
            // against splitting a longer identifier (e.g. token "in" inside "index").
            // The continuation check is widened by any characters registered via
            // Builder.identifierPartExtra(...) (e.g. '[' and ']' so "y[4]" is scanned
            // as one candidate run and handed to the dynamic predicate whole, instead
            // of being truncated at "y" before the predicate ever sees the brackets).
            int identEnd = -1;
            if (Character.isJavaIdentifierStart(currentChar)) {
                identEnd = cursor + 1;
                while (identEnd < length && isIdentifierPart(input.charAt(identEnd))) {
                    identEnd++;
                }
            }

            boolean matched = false;
            if (dynamicMatchFirst) {
                if (tryDynamicMatch(output, literalStart, cursor, identEnd)) {
                    cursor = identEnd;
                    literalStart = cursor;
                    matched = true;
                } else {
                    int consumed = tryStaticMatch(output, literalStart, cursor, length, currentChar, -1);
                    if (consumed > 0) {
                        cursor += consumed;
                        literalStart = cursor;
                        matched = true;
                    }
                }
            } else {
                int consumed = tryStaticMatch(output, literalStart, cursor, length, currentChar, identEnd);
                if (consumed > 0) {
                    cursor += consumed;
                    literalStart = cursor;
                    matched = true;
                } else if (tryDynamicMatch(output, literalStart, cursor, identEnd)) {
                    cursor = identEnd;
                    literalStart = cursor;
                    matched = true;
                }
            }

            if (!matched) {
                if (identEnd != -1) {
                    // The identifier-shaped run didn't satisfy the dynamic matcher and no
                    // static token consumed it either. Skip straight to the end of the run
                    // instead of retrying char-by-char, which would be O(n^2) on long runs
                    // of non-matching letters. It simply accumulates as part of the pending
                    // literal, to be flushed later.
                    cursor = identEnd;
                } else {
                    cursor++;
                }
            }
        }

        flushLiteral(output, literalStart, length);
        cachedResult = output;
        return cachedResult;
    }

    /**
     * Attempts a static token match at {@code cursor}. If {@code identEnd} is not -1
     * (i.e. we're sitting inside an identifier-shaped run), a candidate token is
     * rejected when it would end in the middle of that run, since that would split
     * a longer identifier apart (e.g. token "in" should not fire inside "index").
     *
     * @param literalStart start of the pending literal run, flushed on a successful match.
     * @return number of characters consumed (> 0) if matched, otherwise 0.
     */
    private int tryStaticMatch(List<String> output, int literalStart, int cursor, int length,
                                char currentChar, int identEnd) {
        List<String> candidates = tokensByFirstChar.getOrDefault(currentChar, Collections.emptyList());

        for (String token : candidates) {
            int tokenLen = token.length();
            int tokenEnd = cursor + tokenLen;
            if (tokenLen <= length - cursor && input.regionMatches(cursor, token, 0, tokenLen)) {
                if (identEnd != -1 && tokenEnd < identEnd) {
                    // Would split an identifier run in half — skip this candidate.
                    continue;
                }
                flushLiteral(output, literalStart, cursor);
                if (includeTokensInOutput) {
                    output.add(token);
                }
                return tokenLen;
            }
        }
        return 0;
    }

    /**
     * Attempts a dynamic identifier match at {@code cursor}, given the precomputed
     * end of the identifier-shaped run ({@code identEnd}, or -1 if none).
     *
     * @param literalStart start of the pending literal run, flushed on a successful match.
     * @return true if the predicate accepted the run and it was emitted.
     */
    private boolean tryDynamicMatch(List<String> output, int literalStart, int cursor, int identEnd) {
        if (identEnd == -1) {
            return false;
        }
        String potentialWord = input.substring(cursor, identEnd);
        if (dynamicTokenMatcher.test(potentialWord)) {
            flushLiteral(output, literalStart, cursor);
            output.add(potentialWord);
            return true;
        }
        return false;
    }

    /**
     * True if {@code c} should be treated as continuing an identifier-shaped run:
     * either a standard Java identifier-part character, or one of the extra
     * characters registered via {@link Builder#identifierPartExtra(char...)}.
     * <p>
     * Note this only affects <em>continuation</em>, not the start of a run — a run
     * still must begin with a character satisfying {@link Character#isJavaIdentifierStart}.
     * Extra characters like '[' or ']' can appear inside/at the end of a run but
     * can't kick one off on their own.
     */
    private boolean isIdentifierPart(char c) {
        return Character.isJavaIdentifierPart(c) || extraIdentifierParts.contains(c);
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
        private boolean dynamicMatchFirst = false;
        private Predicate<String> dynamicTokenMatcher = s -> false;
        private final List<String> allTokens = new ArrayList<>();
        private final HashSet<Character> extraIdentifierParts =  new HashSet<>(); 

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

        /**
         * Controls whether the dynamic identifier predicate is tried before static
         * token matching ({@code true}), or after it ({@code false}, the default —
         * matches original behavior).
         * <p>
         * Enable this when your identifiers should take priority over token text
         * that happens to be a prefix of them (e.g. so a predicate-matched name
         * like "internal" isn't chopped up by a static token "in"). Be aware that
         * in this mode, a predicate matching a string identical to one of your
         * static tokens will shadow that token entirely.
         */
        public Builder matchDynamicFirst(boolean dynamicFirst) {
            this.dynamicMatchFirst = dynamicFirst;
            return this;
        }

        /**
         * Widens what counts as part of an identifier-shaped run, beyond standard
         * Java identifier characters (letters, digits, {@code _}, {@code $}).
         * <p>
         * By default, the scanner decides where a candidate run for the dynamic
         * predicate starts and ends using {@link Character#isJavaIdentifierStart}
         * and {@link Character#isJavaIdentifierPart} alone — so for input like
         * {@code "y[4]"}, the run stops right after {@code y}, and the dynamic
         * predicate never even sees the {@code [4]} suffix. Registering
         * {@code '['} and {@code ']'} here extends the run to cover the whole
         * {@code "y[4]"} span so your predicate can validate (or reject) it as
         * a single candidate.
         * <p>
         * Only affects continuation of a run, not what can start one — a run
         * must still begin with a standard Java identifier-start character.
         * <p>
         * Note: if any of these characters are also registered as static tokens
         * (e.g. brackets used for matrix literals elsewhere in your grammar),
         * the static-token boundary guard will defer to the dynamic predicate
         * whenever they appear immediately after an identifier-start character,
         * since consuming them there would otherwise split the identifier run.
         */
        public Builder identifierPartExtra(char... extraChars) {
            if (extraChars != null) {
                for (char c : extraChars) {
                    this.extraIdentifierParts.add(c);
                }
            }
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

        // 3. Boundary-fix demo: static token "in" must NOT split identifier "index"
        String boundaryInput = "index in range";
        String[] boundaryTokens = {"in"};
        Predicate<String> anyLetters = w -> w.chars().allMatch(Character::isLetter);

        Scanner staticFirst = new Scanner.Builder(boundaryInput)
                .ignoreWhitespace(true)
                .withDynamicMatcher(anyLetters)
                .addTokens(boundaryTokens)
                .build();
        System.out.println("Static-first (boundary-safe): " + staticFirst.scan());

        // 4. Dynamic-first demo: identifiers win priority over static tokens entirely
        Scanner dynamicFirst = new Scanner.Builder(boundaryInput)
                .ignoreWhitespace(true)
                .withDynamicMatcher(anyLetters)
                .addTokens(boundaryTokens)
                .matchDynamicFirst(true)
                .build();
        System.out.println("Dynamic-first (identifiers shadow tokens): " + dynamicFirst.scan());

        // 5. identifierPartExtra demo: "y[4]" should be grouped as one variable
        // once '[' and ']' are registered as identifier-continuation characters.
        String subscriptInput = "y[4]+3*x";
        String[] subscriptTokens = {"+", "*"};
        Predicate<String> isSubscriptedVar = w -> w.matches("[a-zA-Z]\\w*(\\[\\d+])?");

        Scanner withoutExtra = new Scanner.Builder(subscriptInput)
                .ignoreWhitespace(true)
                .withDynamicMatcher(isSubscriptedVar)
                .addTokens(subscriptTokens)
                .matchDynamicFirst(true)
                .build();
        System.out.println("Without identifierPartExtra: " + withoutExtra.scan());

        Scanner withExtra = new Scanner.Builder(subscriptInput)
                .ignoreWhitespace(true)
                .withDynamicMatcher(isSubscriptedVar)
                .addTokens(subscriptTokens)
                .matchDynamicFirst(true)
                .identifierPartExtra('[', ']')
                .build();
        System.out.println("With identifierPartExtra('[', ']'): " + withExtra.scan());
    }
}