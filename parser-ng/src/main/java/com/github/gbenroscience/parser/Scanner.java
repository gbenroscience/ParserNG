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
 * static token dictionaries, dynamic identifier matching (via Predicates),
 * generic delimited regions (e.g. quoted strings, block comments), and
 * optional whitespace stripping.
 * <p>
 * <b>Match ordering:</b> delimited regions are always checked first, since they
 * represent a "don't tokenize inside here" zone rather than a token in their own
 * right. After that, static tokens are tried before the dynamic identifier
 * predicate ("static-first") by default. This can be flipped to "dynamic-first"
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
    private final boolean stripDelimiterMarkers;
    private final UnterminatedDelimiterPolicy unterminatedDelimiterPolicy;
    private final Map<Character, List<String>> tokensByFirstChar;
    private final Predicate<String> dynamicTokenMatcher;
    private final HashSet<Character> extraIdentifierParts;
    private final Map<Character, List<Delimiter>> delimitersByFirstChar;

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
        this.stripDelimiterMarkers = builder.stripDelimiterMarkers;
        this.unterminatedDelimiterPolicy = builder.unterminatedDelimiterPolicy;
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

        Map<Character, List<Delimiter>> delimMap = new HashMap<>();
        for (Delimiter d : builder.delimiters) {
            delimMap.computeIfAbsent(d.start.charAt(0), k -> new ArrayList<>()).add(d);
        }
        // Longest start-marker first, so e.g. "\"\"\"" (triple-quote) is preferred
        // over "\"" when both are registered and both match at the same cursor.
        for (List<Delimiter> list : delimMap.values()) {
            list.sort((a, b) -> Integer.compare(b.start.length(), a.start.length()));
        }
        this.delimitersByFirstChar = Collections.unmodifiableMap(delimMap);
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

            // 2. Delimited-region fast-forwarding (quoted strings, block comments, etc.)
            // Deliberately checked before static/dynamic matching: a delimited region
            // means "don't tokenize inside here", so nothing downstream should ever
            // get a look at its interior.
            if (!delimitersByFirstChar.isEmpty()) {
                int consumed = tryDelimitedMatch(output, literalStart, cursor, length, currentChar);
                if (consumed > 0) {
                    cursor += consumed;
                    literalStart = cursor;
                    continue;
                }
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
     * Attempts a delimited-region match at {@code cursor}: if a registered start
     * marker matches here, scans forward for its corresponding end marker
     * (honoring that delimiter's escape character, if any) and emits the whole
     * span — including nested static/dynamic-token-shaped text — as a single
     * opaque literal.
     * <p>
     * If the end marker is never found before the input runs out, the region is
     * treated as unterminated and consumes through the end of the input; this is
     * a deliberate, permissive choice (rather than throwing) so a single stray
     * quote doesn't blow up the whole scan — callers who want strictness can
     * detect this by checking whether the last output token ends with the
     * expected end marker.
     *
     * @return number of characters consumed (> 0) if a delimiter started here, otherwise 0.
     */
    private int tryDelimitedMatch(List<String> output, int literalStart, int cursor, int length, char currentChar) {
        List<Delimiter> candidates = delimitersByFirstChar.getOrDefault(currentChar, Collections.emptyList());
        if (candidates.isEmpty()) {
            return 0;
        }

        for (Delimiter d : candidates) {
            int startLen = d.start.length();
            if (startLen <= length - cursor && input.regionMatches(cursor, d.start, 0, startLen)) {
                int contentStart = cursor + startLen;
                DelimiterEnd end = findDelimiterEnd(d, contentStart, length);

                if (!end.terminated) {
                    switch (unterminatedDelimiterPolicy) {
                        case THROW:
                            throw new UnterminatedDelimiterException(d.start, d.end, cursor);
                        case TREAT_AS_NO_MATCH:
                            // Don't consume anything for this candidate; let the normal
                            // static/dynamic/literal machinery handle the start marker
                            // as ordinary text instead. Try the next candidate delimiter
                            // (if any) before giving up entirely.
                            continue;
                        case CONSUME_TO_END:
                        default:
                            // Fall through to normal emission below, using `end` as-is
                            // (index == length, terminated == false).
                            break;
                    }
                }

                flushLiteral(output, literalStart, cursor);

                int spanStart = stripDelimiterMarkers ? contentStart : cursor;
                int spanEnd = (stripDelimiterMarkers && end.terminated)
                        ? end.index - d.end.length()
                        : end.index;
                output.add(input.substring(spanStart, spanEnd));

                return end.index - cursor;
            }
        }
        return 0;
    }

    /**
     * Scans forward from {@code from} looking for delimiter {@code d}'s end marker,
     * skipping escaped characters.
     */
    private DelimiterEnd findDelimiterEnd(Delimiter d, int from, int length) {
        int endLen = d.end.length();
        int i = from;
        while (i < length) {
            if (d.escapeChar != '\0' && input.charAt(i) == d.escapeChar && i + 1 < length) {
                i += 2; // skip the escaped character, whatever it is
                continue;
            }
            if (endLen <= length - i && input.regionMatches(i, d.end, 0, endLen)) {
                return new DelimiterEnd(i + endLen, true);
            }
            i++;
        }
        return new DelimiterEnd(length, false);
    }

    /** Index immediately after a delimiter's end marker, plus whether it was actually found. */
    private static final class DelimiterEnd {
        final int index;
        final boolean terminated;

        DelimiterEnd(int index, boolean terminated) {
            this.index = index;
            this.terminated = terminated;
        }
    }

    /**
     * What to do when a delimited region's start marker is found but its end
     * marker never appears before the input runs out (e.g. {@code foo("hello}
     * with no closing quote).
     */
    public enum UnterminatedDelimiterPolicy {
        /**
         * Consume through the end of input and emit whatever was found as one
         * token, as if it had been properly closed. Silent and permissive —
         * this is the default, preserving the scanner's original behavior for
         * anyone who registers delimiters without setting a policy explicitly.
         * A single malformed quote can swallow the rest of the input; callers
         * who need to detect this can inspect whether the last emitted token
         * ends with the delimiter's end marker.
         */
        CONSUME_TO_END,

        /**
         * Throw {@link UnterminatedDelimiterException} immediately. Use this
         * when a missing closing marker should be treated as a hard lexical
         * error rather than silently absorbed into a token.
         */
        THROW,

        /**
         * Treat the start marker as if it were never a delimiter at all: don't
         * consume it here, and let normal static/dynamic/literal matching
         * handle that character as ordinary text. The scanner will keep
         * scanning; if the same start marker text appears again later with a
         * proper matching end marker, that later occurrence is still eligible
         * to open a valid region.
         */
        TREAT_AS_NO_MATCH
    }

    /**
     * Thrown by the scanner when {@link UnterminatedDelimiterPolicy#THROW} is
     * configured and a delimited region's start marker is found without a
     * matching end marker before the input ends.
     */
    public static class UnterminatedDelimiterException extends RuntimeException {
        private final String delimiterStart;
        private final String delimiterEnd;
        private final int position;

        UnterminatedDelimiterException(String delimiterStart, String delimiterEnd, int position) {
            super("Unterminated delimited region starting with '" + delimiterStart + "' at position "
                    + position + " (expected closing '" + delimiterEnd + "')");
            this.delimiterStart = delimiterStart;
            this.delimiterEnd = delimiterEnd;
            this.position = position;
        }

        /** The start marker text (e.g. {@code "\""}) whose region was left unclosed. */
        public String getDelimiterStart() {
            return delimiterStart;
        }

        /** The end marker text that was expected but never found. */
        public String getDelimiterEnd() {
            return delimiterEnd;
        }

        /** The input index where the unterminated region's start marker began. */
        public int getPosition() {
            return position;
        }
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
    // DELIMITER
    // =========================================================================

    /**
     * A configurable "don't tokenize inside here" region, bounded by a start
     * marker and an end marker (which may be the same string, as with quotes).
     * Not a plain Java record, to keep the class compatible with older Java
     * targets — this file otherwise avoids Java 16+ syntax.
     */
    private static final class Delimiter {
        final String start;
        final String end;
        final char escapeChar; // '\0' means "no escaping"

        Delimiter(String start, String end, char escapeChar) {
            if (start == null || start.isEmpty()) {
                throw new IllegalArgumentException("Delimiter start marker must be non-empty");
            }
            if (end == null || end.isEmpty()) {
                throw new IllegalArgumentException("Delimiter end marker must be non-empty");
            }
            this.start = start;
            this.end = end;
            this.escapeChar = escapeChar;
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
        private boolean stripDelimiterMarkers = false;
        private UnterminatedDelimiterPolicy unterminatedDelimiterPolicy = UnterminatedDelimiterPolicy.CONSUME_TO_END;
        private Predicate<String> dynamicTokenMatcher = s -> false;
        private final List<String> allTokens = new ArrayList<>();
        private final HashSet<Character> extraIdentifierParts =  new HashSet<>();
        private final List<Delimiter> delimiters = new ArrayList<>();

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

        /**
         * Registers a generic delimited region: everything from {@code start}
         * (inclusive) through the next occurrence of {@code end} (inclusive) is
         * emitted as one opaque token, completely bypassing static and dynamic
         * matching for its interior. No escape character.
         * <p>
         * Use this for anything where "don't tokenize inside here" applies:
         * quoted strings ({@code addDelimitedRegion("\"", "\"")}), block comments
         * ({@code addDelimitedRegion("/*", "*&#47;")}), custom bracketed raw
         * blocks ({@code addDelimitedRegion("{{", "}}")}), etc.
         * <p>
         * If two registered regions could start at the same position, the one
         * with the longer start marker wins (so a triple-quote marker takes
         * priority over a single-quote marker starting with the same character).
         * @param start
         * @param end
         * @return 
         */
        public Builder addDelimitedRegion(String start, String end) {
            return addDelimitedRegion(start, end, '\0');
        }

        /**
         * Like {@link #addDelimitedRegion(String, String)}, but treats
         * {@code escapeChar} as an escape marker inside the region: the character
         * immediately following {@code escapeChar} is always skipped when
         * searching for {@code end}, so it can never prematurely terminate the
         * region. E.g. with {@code addDelimitedRegion("\"", "\"", '\\')}, the
         * input {@code "a \"b\" c"} is consumed as a single token instead of
         * terminating at the escaped inner quote.
         * <p>
         * Pass {@code '\0'} for no escaping (equivalent to the two-arg overload).
         * @param start
         * @param end
         * @param escapeChar
         * @return 
         */
        public Builder addDelimitedRegion(String start, String end, char escapeChar) {
            this.delimiters.add(new Delimiter(start, end, escapeChar));
            return this;
        }

        /**
         * Controls whether the start/end markers themselves are kept in the
         * emitted token ({@code false}, the default — {@code "hello"} is emitted
         * as {@code "hello"} including the quote characters) or stripped
         * ({@code true} — emitted as just {@code hello}).
         * <p>
         * This is a single global switch covering all registered delimited
         * regions, not a per-region setting.
         * @param strip
         * @return 
         */
        public Builder stripDelimiterMarkers(boolean strip) {
            this.stripDelimiterMarkers = strip;
            return this;
        }

        /**
         * Controls what happens when a delimited region's start marker is found
         * but its end marker never appears before the input ends (e.g.
         * {@code foo("hello} with no closing quote).
         * <p>
         * Defaults to {@link UnterminatedDelimiterPolicy#CONSUME_TO_END} — the
         * scanner's original permissive behavior, preserved for anyone who
         * doesn't call this method. Callers who want a missing closing marker
         * to be a hard error should pass {@link UnterminatedDelimiterPolicy#THROW};
         * callers who'd rather the stray start marker just be treated as
         * ordinary text should pass {@link UnterminatedDelimiterPolicy#TREAT_AS_NO_MATCH}.
         * <p>
         * This is a single global policy covering all registered delimited
         * regions, not a per-region setting.
         */
        public Builder onUnterminatedDelimiter(UnterminatedDelimiterPolicy policy) {
            this.unterminatedDelimiterPolicy = policy != null ? policy : UnterminatedDelimiterPolicy.CONSUME_TO_END;
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

        // 6. Delimited-region demo: quoted strings stay intact, even containing
        // characters that are registered as static tokens (',', '+', etc).
        String quotedInput = "foo(\"hello + world\", 'a,b,c') + x";
        String[] quotedTokens = {"foo", "(", ")", ",", "+"};

        Scanner quoteAware = new Scanner.Builder(quotedInput)
                .ignoreWhitespace(true)
                .addTokens(quotedTokens)
                .addDelimitedRegion("\"", "\"")
                .addDelimitedRegion("'", "'")
                .build();
        System.out.println("Quote-aware (markers kept): " + quoteAware.scan());

        Scanner quoteAwareStripped = new Scanner.Builder(quotedInput)
                .ignoreWhitespace(true)
                .addTokens(quotedTokens)
                .addDelimitedRegion("\"", "\"")
                .addDelimitedRegion("'", "'")
                .stripDelimiterMarkers(true)
                .build();
        System.out.println("Quote-aware (markers stripped): " + quoteAwareStripped.scan());

        // 7. Escaped-quote demo: an escaped inner quote must not terminate the region early.
        String escapedInput = "say(\"a \\\"b\\\" c\")";
        Scanner escapeAware = new Scanner.Builder(escapedInput)
                .ignoreWhitespace(true)
                .addTokens(new String[]{"say", "(", ")"})
                .addDelimitedRegion("\"", "\"", '\\')
                .build();
        System.out.println("Escape-aware: " + escapeAware.scan());

        // 8. Unterminated-delimiter policy demo.
        String unterminatedInput = "foo(\"hello + x * y";
        String[] unterminatedTokens = {"foo", "(", ")", "+", "*"};

        Scanner consumeToEnd = new Scanner.Builder(unterminatedInput)
                .ignoreWhitespace(true)
                .addTokens(unterminatedTokens)
                .addDelimitedRegion("\"", "\"")
                .onUnterminatedDelimiter(Scanner.UnterminatedDelimiterPolicy.CONSUME_TO_END) // default
                .build();
        System.out.println("Unterminated, CONSUME_TO_END (default): " + consumeToEnd.scan());

        Scanner treatAsNoMatch = new Scanner.Builder(unterminatedInput)
                .ignoreWhitespace(true)
                .addTokens(unterminatedTokens)
                .addDelimitedRegion("\"", "\"")
                .onUnterminatedDelimiter(Scanner.UnterminatedDelimiterPolicy.TREAT_AS_NO_MATCH)
                .build();
        System.out.println("Unterminated, TREAT_AS_NO_MATCH: " + treatAsNoMatch.scan());

        Scanner throwsOnUnterminated = new Scanner.Builder(unterminatedInput)
                .ignoreWhitespace(true)
                .addTokens(unterminatedTokens)
                .addDelimitedRegion("\"", "\"")
                .onUnterminatedDelimiter(Scanner.UnterminatedDelimiterPolicy.THROW)
                .build();
        try {
            throwsOnUnterminated.scan();
        } catch (Scanner.UnterminatedDelimiterException e) {
            System.out.println("Unterminated, THROW: caught -> " + e.getMessage());
        }
    }
}