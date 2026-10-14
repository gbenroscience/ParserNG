/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.math.otherBaseParser;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Number-base support for {@link BaseNFunction}: parsing and printing numbers in
 * any base from 2 to 36, and rewriting an expression whose numbers are written
 * in such a base into the equivalent decimal expression.
 * <p>
 * Digits are {@code 0-9} followed by {@code A-Z} (case-insensitive), so base 16
 * uses {@code 0-9A-F} and base 36 uses {@code 0-9A-Z}. Values are carried as
 * {@code double}s, exactly like {@link com.github.gbenroscience.parser.MathExpression}
 * does, so precision is that of a double.
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public final class BaseNumbers {

    public static final int MIN_BASE = 2;
    public static final int MAX_BASE = 36;

    /**
     * Hard limit on the number of fractional digits produced for one number
     * (a double can need up to 1074 binary digits).
     */
    private static final int MAX_FRACTION_DIGITS = 1100;

    private BaseNumbers() {
    }

    /**
     * @param base the base to validate
     * @return {@code base}
     * @throws IllegalArgumentException if it is outside 2..36
     */
    public static int checkBase(int base) {
        if (base < MIN_BASE || base > MAX_BASE) {
            throw new IllegalArgumentException("Unsupported number base " + base
                    + ". Bases from " + MIN_BASE + " to " + MAX_BASE + " are supported.");
        }
        return base;
    }

    /**
     * @return the value of an ASCII digit or letter ({@code 0-9}, {@code A-Z},
     * {@code a-z} -> 0..35), or -1 for anything else.
     */
    private static int digitValue(char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'A' && c <= 'Z') {
            return c - 'A' + 10;
        }
        if (c >= 'a' && c <= 'z') {
            return c - 'a' + 10;
        }
        return -1;
    }

    private static boolean isAsciiDigit(char c) {
        return c >= '0' && c <= '9';
    }

    // ------------------------------------------------------------------
    //  Parsing
    // ------------------------------------------------------------------
    /**
     * Parses a number written in {@code base}: an optional sign, digits, and an
     * optional fractional part after a point, e.g. {@code -1A.8} in base 16.
     * {@code NaN}, {@code Infinity} and {@code -Infinity} are accepted too, so
     * that anything produced by {@link #format(double, int)} can be read back.
     *
     * @param text the number
     * @param base 2..36
     * @return the value
     * @throws NumberFormatException if the text is not a number in that base
     */
    public static double parse(String text, int base) {
        checkBase(base);
        Objects.requireNonNull(text, "text");
        String s = text.trim();
        if (s.equals("NaN")) {
            return Double.NaN;
        }
        if (s.equals("Infinity") || s.equals("+Infinity")) {
            return Double.POSITIVE_INFINITY;
        }
        if (s.equals("-Infinity")) {
            return Double.NEGATIVE_INFINITY;
        }
        int i = 0;
        final int n = s.length();
        boolean negative = false;
        if (i < n && (s.charAt(i) == '-' || s.charAt(i) == '+')) {
            negative = s.charAt(i) == '-';
            i++;
        }
        int wholeStart = i;
        while (i < n && s.charAt(i) != '.') {
            requireDigit(s, i, base, text);
            i++;
        }
        String whole = s.substring(wholeStart, i);
        String frac = "";
        if (i < n) { // at the point
            frac = s.substring(i + 1);
            for (int k = 0; k < frac.length(); k++) {
                requireDigit(frac, k, base, text);
            }
        }
        if (whole.isEmpty() && frac.isEmpty()) {
            throw new NumberFormatException("Not a base-" + base + " number: \"" + text + "\"");
        }

        BigDecimal value;
        if (whole.isEmpty()) {
            value = BigDecimal.ZERO;
        } else if (whole.length() <= 10) {
            value = BigDecimal.valueOf(Long.parseLong(whole, base));
        } else {
            value = new BigDecimal(new BigInteger(whole, base));
        }
        if (!frac.isEmpty()) {
            // Horner from the least significant digit: ((d_n/b + d_{n-1})/b + ...)/b
            BigDecimal b = BigDecimal.valueOf(base);
            BigDecimal f = BigDecimal.ZERO;
            for (int k = frac.length() - 1; k >= 0; k--) {
                f = f.add(BigDecimal.valueOf(digitValue(frac.charAt(k)))).divide(b, MathContext.DECIMAL128);
            }
            value = value.add(f);
        }
        double d = value.doubleValue();
        return negative ? -d : d;
    }

    private static void requireDigit(String s, int index, int base, String original) {
        char c = s.charAt(index);
        int v = digitValue(c);
        if (v < 0 || v >= base) {
            throw new NumberFormatException("'" + c + "' is not a valid digit in base " + base
                    + " (in \"" + original + "\")");
        }
    }

    // ------------------------------------------------------------------
    //  Formatting
    // ------------------------------------------------------------------
    /**
     * Writes a value in {@code base}. The result always has a point, like
     * {@code MathExpression} output ({@code 11.0}), uses upper-case letters for
     * digits above 9, and carries as many fractional digits as a double can
     * justify (about 53 bits worth), with trailing zeros removed.
     * {@code NaN} and the infinities are returned as {@code NaN},
     * {@code Infinity} and {@code -Infinity}.
     *
     * @param value the value
     * @param base 2..36
     * @return the text
     */
    public static String format(double value, int base) {
        checkBase(base);
        if (Double.isNaN(value)) {
            return "NaN";
        }
        if (Double.isInfinite(value)) {
            return value > 0 ? "Infinity" : "-Infinity";
        }
        if (value == 0) {
            return "0.0";
        }
        boolean negative = value < 0;
        double abs = Math.abs(value);

        if (base == 10) {
            // Shortest decimal that round-trips: what Double.toString shows, but never in
            // E-notation.
            String plain = new BigDecimal(Double.toString(abs)).toPlainString();
            if (plain.indexOf('.') < 0) {
                plain += ".0";
            }
            return negative ? "-" + plain : plain;
        }

        BigDecimal exact = new BigDecimal(abs); // the double's exact value
        BigInteger whole = exact.toBigInteger();
        BigDecimal frac = exact.subtract(new BigDecimal(whole));

        // A double is a binary fraction, so in base 2, 4, 8, 16 and 32 its expansion is finite:
        // print all of it, which makes the text exact. In any other base the expansion repeats, so
        // keep the significant digits that 53 bits can back up and round there (this is what makes
        // 1.0/3 print as 0.1 in base 3 rather than as a long tail of 2s).
        final boolean finiteExpansion = (base & (base - 1)) == 0;
        final int budget = finiteExpansion ? Integer.MAX_VALUE - 1
                : Math.max(1, (int) Math.floor(53.0 * Math.log(2) / Math.log(base)));

        List<Integer> digits = new ArrayList<>();
        boolean significant = whole.signum() != 0;
        int sig = significant ? whole.toString(base).length() : 0;
        BigDecimal b = BigDecimal.valueOf(base);
        // generate budget digits plus one guard digit used for rounding
        while (frac.signum() != 0 && digits.size() < MAX_FRACTION_DIGITS && !(significant && sig >= budget + 1)) {
            frac = frac.multiply(b);
            int d = frac.toBigInteger().intValue();
            frac = frac.subtract(BigDecimal.valueOf(d));
            digits.add(d);
            if (d != 0) {
                significant = true;
            }
            if (significant) {
                sig++;
            }
        }
        if (significant && sig >= budget + 1 && !digits.isEmpty()) {
            int guard = digits.remove(digits.size() - 1);
            if (guard * 2 >= base) { // round half up, carrying through the digits
                int k = digits.size() - 1;
                while (k >= 0) {
                    int v = digits.get(k) + 1;
                    if (v == base) {
                        digits.set(k, 0);
                        k--;
                    } else {
                        digits.set(k, v);
                        break;
                    }
                }
                if (k < 0) {
                    whole = whole.add(BigInteger.ONE);
                }
            }
        }
        // trailing zeros carry no information
        int last = digits.size();
        while (last > 0 && digits.get(last - 1) == 0) {
            last--;
        }

        StringBuilder sb = new StringBuilder(whole.toString(base).toUpperCase(Locale.ROOT));
        sb.append('.');
        if (last == 0) {
            sb.append('0');
        } else {
            for (int k = 0; k < last; k++) {
                sb.append(Character.toUpperCase(Character.forDigit(digits.get(k), base)));
            }
        }
        return negative ? "-" + sb : sb.toString();
    }

    /**
     * Rewrites the text a {@code MathExpression} printed (a number, or a list such
     * as {@code [1.0, 2.5]}) so that its numbers are in {@code base}. Anything
     * that is not a number or a list of numbers is returned unchanged.
     */
    static String convertResult(String result, int base) {
        if (result == null || base == 10) {
            return result;
        }
        String r = result.trim();
        if (r.length() > 1 && r.charAt(0) == '[' && r.charAt(r.length() - 1) == ']') {
            String inner = r.substring(1, r.length() - 1).trim();
            if (inner.isEmpty()) {
                return result;
            }
            String[] parts = inner.split(",");
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < parts.length; i++) {
                Double d = tryDouble(parts[i].trim());
                if (d == null) {
                    return result; // not a plain list of numbers
                }
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(format(d, base));
            }
            return sb.append(']').toString();
        }
        Double d = tryDouble(r);
        return d == null ? result : format(d, base);
    }

    private static Double tryDouble(String s) {
        if (s.isEmpty()) {
            return null;
        }
        // Double.parseDouble also accepts forms like "5d", "0x1p3" and padded text, none of which a
        // MathExpression result looks like, so insist on a plain numeric shape first.
        char c = s.charAt(0);
        if (!(isAsciiDigit(c) || c == '-' || c == '+' || c == '.' || c == 'N' || c == 'I')) {
            return null;
        }
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (!(isAsciiDigit(ch) || ch == '.' || ch == '-' || ch == '+' || ch == 'E' || ch == 'e'
                    || ch == 'N' || ch == 'a' || ch == 'I' || ch == 'n' || ch == 'f' || ch == 'i' || ch == 't' || ch == 'y')) {
                return null;
            }
        }
        try {
            return Double.parseDouble(s);
        } catch (NumberFormatException ex) {
            return null;
        }
    }

    // ------------------------------------------------------------------
    //  Expression rewriting
    // ------------------------------------------------------------------
    /**
     * Rewrites an expression so that every number literal, which is written in
     * {@code base}, becomes the equivalent decimal literal. Everything else
     * (names, operators, brackets, string literals) is left alone, so the result
     * is an ordinary decimal expression for {@code MathExpression}.
     * <p>
     * Rules:
     * <ul>
     * <li>A number literal starts with a digit {@code 0-9} (or a point followed by
     * a digit). Anything starting with a letter, {@code _} or {@code $} is a name,
     * so a hexadecimal value that begins with a letter needs a leading zero:
     * write {@code 0FF}, not {@code FF}.</li>
     * <li>Names may contain digits ({@code x2}, {@code log10}); those are never
     * converted.</li>
     * <li>The sizes in a matrix or vector literal, such as the {@code 2,2} of
     * {@code @(2,2)(1,2,3,4)}, are plain decimal counts and are kept.</li>
     * <li>The index in {@code x[2]} is a plain decimal and is kept.</li>
     * <li>Up to base 10, letters are never digits, so {@code 11x} still means
     * {@code 11*x}. Above base 10, letters that are valid digits belong to the
     * literal, so a literal directly followed by another letter ({@code 2abs(x)}
     * in base 16) is ambiguous and is rejected; write {@code 2*abs(x)}.</li>
     * <li>Scientific notation ({@code 1E5}) is not available outside base 10.</li>
     * </ul>
     * Base 10 returns the text untouched.
     *
     * @param input the expression, with numbers in {@code base}
     * @param base 2..36
     * @return the decimal expression
     * @throws NumberFormatException if a literal has a digit that is not valid in
     * {@code base}, or is ambiguous
     */
    public static String toDecimalExpression(String input, int base) {
        checkBase(base);
        Objects.requireNonNull(input, "input");
        if (base == 10) {
            return input;
        }
        final int n = input.length();
        StringBuilder out = new StringBuilder(n + 16);
        int i = 0;
        while (i < n) {
            char c = input.charAt(i);

            if (c == '"' || c == '\'') { // string literal: copy through the closing quote
                int j = i + 1;
                while (j < n && input.charAt(j) != c) {
                    if (input.charAt(j) == '\\') {
                        j++;
                    }
                    j++;
                }
                j = Math.min(j + 1, n);
                out.append(input, i, j);
                i = j;
                continue;
            }

            if (Character.isLetter(c) || c == '_' || c == '$') { // a name
                int j = i + 1;
                while (j < n && (Character.isLetterOrDigit(input.charAt(j)) || input.charAt(j) == '_' || input.charAt(j) == '$')) {
                    j++;
                }
                if (j < n && input.charAt(j) == '[') { // x[2]: the index is decimal
                    int close = input.indexOf(']', j);
                    j = close < 0 ? n : close + 1;
                }
                out.append(input, i, j);
                i = j;
                continue;
            }

            if (c == '@' && i + 1 < n && input.charAt(i + 1) == '(') {
                int close = input.indexOf(')', i);
                if (close > 0 && onlyDigitsAndCommas(input, i + 2, close)) { // @(2,2): decimal sizes
                    out.append(input, i, close + 1);
                    i = close + 1;
                    continue;
                }
            }

            if (isAsciiDigit(c) || (c == '.' && i + 1 < n && isAsciiDigit(input.charAt(i + 1)))) {
                int end = scanLiteral(input, i, base);
                out.append(toDecimalText(input.substring(i, end), base));
                i = end;
                continue;
            }

            out.append(c);
            i++;
        }
        return out.toString();
    }

    private static boolean onlyDigitsAndCommas(String s, int from, int to) {
        if (from >= to) {
            return false;
        }
        for (int i = from; i < to; i++) {
            char c = s.charAt(i);
            if (!(isAsciiDigit(c) || c == ',' || c == ' ')) {
                return false;
            }
        }
        return true;
    }

    /**
     * @return the index just past the number literal that starts at {@code start}
     */
    private static int scanLiteral(String s, int start, int base) {
        final int n = s.length();
        int j = scanDigits(s, start, base);
        if (j < n && s.charAt(j) == '.' && j + 1 < n && digitValue(s.charAt(j + 1)) >= 0
                && (isAsciiDigit(s.charAt(j + 1)) || digitValue(s.charAt(j + 1)) < base)) {
            j = scanDigits(s, j + 1, base);
        }
        if (j < n) {
            char next = s.charAt(j);
            boolean exponent = (next == 'e' || next == 'E') && j + 1 < n
                    && (isAsciiDigit(s.charAt(j + 1)) || ((s.charAt(j + 1) == '+' || s.charAt(j + 1) == '-')
                    && j + 2 < n && isAsciiDigit(s.charAt(j + 2))));
            if (exponent) {
                throw new NumberFormatException("Scientific notation is not supported in base " + base
                        + " (at \"" + s.substring(start, Math.min(n, j + 3)) + "\")");
            }
            if (base > 10 && (Character.isLetterOrDigit(next) || next == '_' || next == '$')) {
                int k = j;
                while (k < n && (Character.isLetterOrDigit(s.charAt(k)) || s.charAt(k) == '_' || s.charAt(k) == '$')) {
                    k++;
                }
                throw new NumberFormatException("Ambiguous number \"" + s.substring(start, k) + "\" in base " + base
                        + ": write * between a number and a name (e.g. 2*abs(x)).");
            }
        }
        return j;
    }

    /**
     * Consumes digits of {@code base} starting at {@code from}. A decimal digit
     * that is too large for the base is an error; a letter that is not a digit of
     * the base simply ends the run.
     */
    private static int scanDigits(String s, int from, int base) {
        int j = from;
        final int n = s.length();
        while (j < n) {
            char c = s.charAt(j);
            if (isAsciiDigit(c)) {
                if (c - '0' >= base) {
                    throw new NumberFormatException("'" + c + "' is not a valid digit in base " + base
                            + " (in \"" + s.substring(from, Math.min(n, j + 1)) + "\")");
                }
                j++;
            } else if (base > 10 && digitValue(c) >= 0 && digitValue(c) < base) {
                j++;
            } else {
                break;
            }
        }
        return j;
    }

    /**
     * @return the decimal text of a base-{@code base} literal, in a form a
     * {@code MathExpression} reads back as the same double and never in E-notation
     */
    private static String toDecimalText(String literal, int base) {
        double v = parse(literal, base);
        return new BigDecimal(Double.toString(v)).toPlainString();
    }
}
