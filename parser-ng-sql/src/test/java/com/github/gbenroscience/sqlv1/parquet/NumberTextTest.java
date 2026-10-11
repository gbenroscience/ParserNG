package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Strict parsing of numbers from string bytes. */
class NumberTextTest {

    private static double pi(String s, NumberText.IntType t) {
        byte[] b = s.getBytes();
        return NumberText.parseInteger(b, b.length, t, "CAST(c AS " + t + ")");
    }

    private static double pd(String s) {
        byte[] b = s.getBytes();
        return NumberText.parseDouble(b, b.length, "CAST(c AS DOUBLE)");
    }

    private static void assertBad(Runnable r, String fragment) {
        ParquetSqlException e = assertThrows(ParquetSqlException.class, r::run);
        assertTrue(e.getMessage().contains(fragment), e.getMessage());
    }

    @Test
    void integersParseWithSignsAndSpaces() {
        Random r = new Random(2);
        for (int i = 0; i < 20_000; i++) {
            long v = r.nextLong() >> r.nextInt(60);
            if (Math.abs(v) > (1L << 53)) continue;
            String s = (r.nextBoolean() ? " " : "") + (v >= 0 && r.nextBoolean() ? "+" : "") + v + (r.nextBoolean() ? "  " : "");
            assertEquals((double) v, pi(s, NumberText.IntType.BIGINT), s);
        }
        assertEquals(-3, pi("-3", NumberText.IntType.INTEGER));
        assertEquals(5, pi("+5", NumberText.IntType.SMALLINT));
        assertEquals(7, pi("  7 ", NumberText.IntType.INTEGER));
    }

    @Test
    void integerRangesAndTheFloat8LimitAreEnforced() {
        assertEquals(2147483647.0, pi("2147483647", NumberText.IntType.INTEGER));
        assertEquals(-2147483648.0, pi("-2147483648", NumberText.IntType.INTEGER));
        assertBad(() -> pi("2147483648", NumberText.IntType.INTEGER), "out of range for INTEGER");
        assertBad(() -> pi("32768", NumberText.IntType.SMALLINT), "out of range for SMALLINT");
        assertBad(() -> pi("9007199254740993", NumberText.IntType.BIGINT), "beyond 2^53");
        assertBad(() -> pi("99999999999999999999", NumberText.IntType.BIGINT), "out of range for BIGINT");
    }

    @Test
    void anythingThatIsNotAPlainIntegerIsRejectedWithTheValue() {
        for (String s : new String[] {"", "  ", "abc", "12.7", "1e3", "0x10", "--1", "+", "1 2", "12a"}) {
            assertBad(() -> pi(s, NumberText.IntType.INTEGER), "not a valid integer");
        }
        assertBad(() -> pi("xyz", NumberText.IntType.INTEGER), "CAST(c AS INTEGER): 'xyz'");
    }

    @Test
    void doublesRoundTripAndAcceptTheUsualForms() {
        Random r = new Random(3);
        for (int i = 0; i < 20_000; i++) {
            double v = (r.nextDouble() - 0.5) * Math.pow(10, r.nextInt(40) - 20);
            assertEquals(v, pd(Double.toString(v)));
            assertEquals(v, pd(" " + v + " "));
        }
        assertEquals(0.5, pd(".5"));
        assertEquals(5.0, pd("5."));
        assertEquals(1000.0, pd("1e3"));
        assertEquals(0.01, pd("1E-2"));
        assertEquals(300.0, pd("+3.0e+2"));
        assertEquals(42.0, pd("  42  "));
    }

    @Test
    void doublesRejectWhatJavaWouldQuietlyAccept() {
        for (String s : new String[] {"", "abc", "NaN", "Infinity", "-Infinity", "1e", "1e+", "0x1p3", "1.5d", "1.5f", "1,5", "1 5", "--1", ".", "+", "e5"}) {
            assertBad(() -> pd(s), "not a valid number");
        }
        assertBad(() -> pd("1e999"), "out of range for DOUBLE");
    }
}
