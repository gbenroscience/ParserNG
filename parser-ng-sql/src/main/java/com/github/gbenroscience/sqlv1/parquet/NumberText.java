package com.github.gbenroscience.sqlv1.parquet;

import java.nio.charset.StandardCharsets;

/**
 * Strict number parsing from the ASCII bytes of a string value, for {@code CAST(string AS number)}. No
 * Arrow dependency. Surrounding spaces are ignored; anything else that is not a plain number is an error
 * that names the expression and the value (never a silent {@code 0} or {@code NaN}).
 */
final class NumberText {

    private NumberText() { }

    private static final long TWO_POW_53 = 1L << 53;

    /** The integer targets of {@code CAST}, with their ranges. */
    enum IntType {
        SMALLINT(Short.MIN_VALUE, Short.MAX_VALUE), INTEGER(Integer.MIN_VALUE, Integer.MAX_VALUE), BIGINT(Long.MIN_VALUE, Long.MAX_VALUE);

        final long min, max;

        IntType(long min, long max) {
            this.min = min;
            this.max = max;
        }
    }

    /**
     * {@code [+-]digits} as an integer of {@code type}. The result is a {@code double} (the engine's type), so a
     * value beyond 2^53 is rejected instead of rounded.
     */
    static double parseInteger(byte[] b, int len, IntType type, String text) {
        int a = 0, e = len;
        while (a < e && b[a] == ' ') a++;
        while (e > a && b[e - 1] == ' ') e--;
        int p = a;
        boolean neg = false;
        if (p < e && (b[p] == '-' || b[p] == '+')) {
            neg = b[p] == '-';
            p++;
        }
        if (p >= e) throw bad(b, len, text, "is not a valid integer");
        long v = 0;
        boolean overflow = false;
        for (; p < e; p++) {
            int d = b[p] - '0';
            if (d < 0 || d > 9) throw bad(b, len, text, "is not a valid integer");
            if (v > (Long.MAX_VALUE - d) / 10) overflow = true;
            else v = v * 10 + d;
        }
        if (overflow) throw bad(b, len, text, "is out of range for " + type);
        long r = neg ? -v : v;
        if (r < type.min || r > type.max) throw bad(b, len, text, "is out of range for " + type);
        if (Math.abs(r) > TWO_POW_53) {
            throw bad(b, len, text, "is beyond 2^53, which a Float8 result cannot hold exactly");
        }
        return r;
    }

    /** {@code [+-](digits[.digits] | .digits)[(e|E)[+-]digits]}: no {@code NaN}, no {@code Infinity}, no hex, no suffixes. */
    static double parseDouble(byte[] b, int len, String text) {
        int a = 0, e = len;
        while (a < e && b[a] == ' ') a++;
        while (e > a && b[e - 1] == ' ') e--;
        int p = a;
        if (p < e && (b[p] == '-' || b[p] == '+')) p++;
        int intDigits = 0, fracDigits = 0;
        while (p < e && b[p] >= '0' && b[p] <= '9') { p++; intDigits++; }
        if (p < e && b[p] == '.') {
            p++;
            while (p < e && b[p] >= '0' && b[p] <= '9') { p++; fracDigits++; }
        }
        if (intDigits + fracDigits == 0) throw bad(b, len, text, "is not a valid number");
        if (p < e && (b[p] == 'e' || b[p] == 'E')) {
            p++;
            if (p < e && (b[p] == '-' || b[p] == '+')) p++;
            int exp = 0;
            while (p < e && b[p] >= '0' && b[p] <= '9') { p++; exp++; }
            if (exp == 0) throw bad(b, len, text, "is not a valid number");
        }
        if (p != e) throw bad(b, len, text, "is not a valid number");
        double d = Double.parseDouble(new String(b, a, e - a, StandardCharsets.US_ASCII));
        if (Double.isInfinite(d)) throw bad(b, len, text, "is out of range for DOUBLE");
        return d;
    }

    private static ParquetSqlException bad(byte[] b, int len, String text, String why) {
        return new ParquetSqlException(text + ": '" + new String(b, 0, len, StandardCharsets.UTF_8) + "' " + why);
    }
}
