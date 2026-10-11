package com.github.gbenroscience.sqlv1;

import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Delimited identifiers ({@code "my col"} and {@code `my col`}).
 *
 * <p>The parser keeps expressions as text that is later compiled by ParserNG, which only knows plain
 * identifiers. A quoted identifier that is already a plain identifier ({@code "id"}, not a keyword) is
 * written as that identifier. Any other ({@code "my col"}, {@code "1st"}, {@code "select"},
 * {@code "naïve"}) is written as a <em>reference</em>: {@code pngq}, the UTF-8 bytes of the name as two
 * letters each ({@code a}..{@code p} for the nibble), then {@code z}. A reference is letters only, so
 * ParserNG and every identifier scan in this project treat it as an ordinary variable, and it carries
 * its own meaning, so no side table travels with the statement. {@link #decode} turns it back into
 * the real name.
 *
 * <p>Only {@code ParquetSql} resolves references; {@code ArrowSql} sees an unknown column name.
 */
public final class QuotedIdentifiers {

    private static final String PREFIX = "pngq";
    private static final char SUFFIX = 'z';
    private static final Pattern REF = Pattern.compile("\\bpngq(?:[a-p]{2})+z\\b");
    private static final Pattern PLAIN = Pattern.compile("[A-Za-z][A-Za-z0-9_]*");

    private QuotedIdentifiers() { }

    /** The text that stands for the quoted identifier {@code name} inside an expression. */
    public static String reference(String name) {
        if (PLAIN.matcher(name).matches() && !SqlLexer.isKeyword(name)) {
            return name;
        }
        return encode(name);
    }

    /** Always the encoded form, whatever the name. */
    public static String encode(String name) {
        byte[] b = name.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder(PREFIX.length() + 2 * b.length + 1).append(PREFIX);
        for (byte x : b) {
            sb.append((char) ('a' + ((x >> 4) & 0xF))).append((char) ('a' + (x & 0xF)));
        }
        return sb.append(SUFFIX).toString();
    }

    /** @return the real name if {@code ref} is exactly an encoded reference, else {@code null} */
    public static String decode(String ref) {
        if (ref == null || ref.length() < PREFIX.length() + 3 || !ref.startsWith(PREFIX)
                || ref.charAt(ref.length() - 1) != SUFFIX || ((ref.length() - PREFIX.length() - 1) & 1) != 0) {
            return null;
        }
        int n = (ref.length() - PREFIX.length() - 1) / 2;
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            char hi = ref.charAt(PREFIX.length() + 2 * i), lo = ref.charAt(PREFIX.length() + 2 * i + 1);
            if (hi < 'a' || hi > 'p' || lo < 'a' || lo > 'p') return null;
            b[i] = (byte) (((hi - 'a') << 4) | (lo - 'a'));
        }
        String name = new String(b, StandardCharsets.UTF_8);
        return encode(name).equals(ref) ? name : null; // reject anything that is not a clean round trip
    }

    /** The real name for {@code ref}, or {@code ref} itself if it is not an encoded reference. */
    public static String real(String ref) {
        String d = decode(ref);
        return d != null ? d : ref;
    }

    /** True if {@code text} contains at least one encoded reference. */
    public static boolean containsReference(String text) {
        return text != null && text.indexOf(PREFIX) >= 0 && REF.matcher(text).find();
    }

    /** {@code text} with every encoded reference replaced by the real name (used for default output names). */
    public static String decodeAll(String text) {
        if (!containsReference(text)) return text;
        Matcher m = REF.matcher(text);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            String d = decode(m.group());
            m.appendReplacement(sb, Matcher.quoteReplacement(d != null ? d : m.group()));
        }
        return m.appendTail(sb).toString();
    }
}
